# UK DIQ vs LEGO — investigation findings & integration decision (2026-06-18)

## Goal
Beat the UK LEGO benchmark (`prediction_rf`) at **GTIN level** on accuracy (1−WAPE) and an
**asymmetric bias** rule (over-forecast fine; under-forecast only OK within 2pp of LEGO), per
category, across rolling backtest origins and the production week **202624 (lag-1)**.

## Proper evaluation methodology (use this for any future comparison)
- **Universe** = keys with `exclude_forecast==0` (the in-scope book). The flag lives in **DT**
  (and the input cache), per `(key, year_week)`. Take keys in-scope on the forward weeks.
- **Benchmark (LEGO)** = `prediction_rf` from `TF_pristine_allkeys` (UC). Where it's null on an
  in-scope key, LEGO effectively = 0 (it leaves ~38% of keys unforecast). The client's shipped
  LEGO (`PristineTVF`) = `prediction_rf` ∪ `ml_pred_final` (DIQ) gap-fill.
- **DIQ** = `ml_pred_final` (prod) or our engine's pooled blend.
- **Actuals** = realized (`actual_sales` / planner `ActualShipment`), matched at key level.
- Metric: GTIN-grain pooled `acc = 1−Σ|f−a|/Σa`, `bias = Σ(f−a)/Σa`.

## What LEGO actually is (read from Databricks `NB_ForwardRun` + LEGO-MODULES-NEW v1.5)
- **Per-key RandomForest** (`groupby(key).applyInPandas`, one RF/key), recursive single-step,
  trained on **outlier-capped `smooth_sales`**, with rich features (STL seasonality indices,
  holiday/promo COEF encoders, brand×APG momentum, weather, price). `XGB=False` → `prediction_rf`.
- Sparse/dead/short keys → **`exclude_forecast` gate → forced to 0** (SBA is shipped but OFF).
- Strong-MOQ rounding for order-quantized segments; forecast cap `[0, 2·hist_max]`.
- Prod assembly (`UK_DIQ_RUN_PRE_POST_SCRIPTS/PostProcess`): final = `mix·DIQ + (1−mix)·prediction_rf`
  for DEO/HOME only, within a window, **mix ≤ 0.25**; pure `prediction_rf` elsewhere; gap zeroed.
- Detail: see memory `lego-model-internals.md`; raw dumps `/tmp/lego_deepdive/`.

## Validated result
- **On the rolling backtest (origins 202615–618, head-to-head GTIN): our pooled DIQ beats LEGO**
  ~0.642 vs 0.625, winning 8/10 categories. Robust across origins.
- **At 202624 (a high-demand spike week): LEGO wins** (0.716 vs our best ~0.69). Everything
  under-forecasts the spike; LEGO under-forecasts least. This single week is LEGO's, consistently.

## What was tried and RULED OUT (none robustly beat the pooled engine)
| Approach | Verdict |
|---|---|
| Per-category stacked ensemble [g,l,rf] | Fails OOS — LEGO's bias flips sign week-to-week. |
| Hierarchical reconciliation (key/GTIN/APG/cat) | Solved bias-robustness but didn't beat LEGO accuracy. |
| In-scope (`exclude_forecast==0`) training | ≈ neutral vs full-universe (no net gain). |
| Feature-port: LEGO FS features → our pooled GBM | **Neutral** (pooled already encodes the signal). |
| Target = capped `smooth_sales` | Regime trade: helps normal weeks, **hurts the 202624 spike** (under-shoots). |
| Per-key RF (LEGO's mechanism) | Helps 202624 (+2.2pp, better bias) but loses normal weeks; still < LEGO at 202624; major engine complexity. |
| Gap-fill hybrid (LEGO + DIQ fill) | = reconstructing `PristineTVF` (LEGO already gap-fills). |

**Root cause:** LEGO's edge at GTIN is the **per-key fit** (key-specific level/seasonality), not
portable features. Our pooled GBM matches it on average but can't fully match per-key on the
spike week. Beating that last mile needs replicating LEGO's full per-key pipeline (diminishing
returns for ~2pp on one week) or genuinely new signal.

## Integration decision (user: "pooled DIQ only")
Ship the **existing pooled, config-driven engine** (`utils/uk_engine.py` + `config/config_uk_base_local.yaml`,
UK; `config_de_base_local.yaml`, DE; market switch in `utils/diq_runner.py::_load_market_io`).
No per-key path, no FS-feature join, no `smooth_sales` target — all ruled out. UK and DE differ by
configuration only.

## Forward options (if revisited)
1. Replicate LEGO's full per-key pipeline (smooth per-key target + full features + tuning) to chase
   the 202624 spike week.
2. New external/causal signal (the only lever that could move accuracy beyond LEGO).
3. Operational: get the production blend to use our pooled DIQ more broadly than DEO/HOME ≤25%
   (a PostProcess change, outside the engine).
