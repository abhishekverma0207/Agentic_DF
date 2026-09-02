# Databricks notebook source
# =============================================================================
# DIQ_FitWeights — cross-market stacked-DIQ FIT (weight estimation), Databricks-only
# =============================================================================
# Market-parametrized: defaults are DE; switch to UK via the "Market presets" cell
# below (the FIT engine + this notebook are generic — only widget values change).
# Modeled on LEGO_RUN_SCRIPT_DE. Estimates the per-category ensemble weights
# [w_global, w_local, w_lego] that the LIVE forward run applies. Re-run this
# notebook whenever the weights need refreshing. Everything runs ON the cluster
# (the /Volumes/ paths are FUSE-mounted → plain pandas/pyarrow reads & writes).
#
# DEPENDENCY CHAIN per snapshot S (what must exist before we can combine/fit):
#       LEGO run  ──►  DT/, FS/  (panel inputs)  +  TF_pristine_allkeys.prediction_xgb (benchmark)
#                          │
#                          ▼
#       new DIQ (compute_components, INLINE) ──► g, l, rf=prediction_xgb  (saved per snapshot)
#                          │
#                          ▼
#       FIT (after all snapshots) ──► join realized actuals ──► per-cat [w_g, w_l, w_xgb] ──► YAML
#   The new DIQ CONSUMES DT⋈FS + prediction_xgb, so LEGO must exist FIRST. This notebook does
#   NOT contain the LEGO pipeline (it's market-specific + needs point-in-time raw data); it
#   ORCHESTRATES — reuse if present, else delegate to the LEGO entry notebook (see STAGE A).
#
# DESIGN — idempotent, requirements-checked per snapshot:
#
#   STAGE 0  discover the snapshot weeks under PARENT_DIR and check, for each, whether
#            the LEGO requirements are already satisfied (DT/, FS/, and TF_pristine_allkeys
#            with a populated prediction_xgb). Also check the actuals snapshot covers each
#            snapshot's forward window. Prints a readiness table.
#
#   STAGE A  for each fit snapshot S:
#            - if LEGO is ALREADY present  → SKIP the LEGO pipeline, run only the new DIQ.
#            - if LEGO is MISSING           → trigger LEGO first via dbutils.notebook.run on a
#              PARAMETRIZED LEGO entry (DE: DE_AllCAT_PARALLEL_RUN; UK: forward-run entry),
#              passing {history_cut_off=S-1, snapshot_week=S, demandiq_run_flag=false} so it
#              runs DT→FE→FS→TF→Combine (producing DT/FS + prediction_xgb) but SKIPS the old
#              per-cat DIQ job fan-out. Off by default (RUN_LEGO_IF_MISSING) — else raises with
#              guidance. Then re-check and proceed.
#            The "new DIQ" = uk_engine.compute_components run INLINE (global g + per-cat l
#            GBMs + LEGO rf=prediction_xgb) over the DT⋈FS panel (built exactly like the
#            LIVE runner) — NOT the old DIQ Run cell. Its 13 forward-week components are SAVED
#            to {COMPONENTS_OUT}/components_S.parquet.
#
#   STAGE B  combine every saved component frame + realized forward actuals (from the
#            ACTUALS_SNAPSHOT DT), join on (key, year_week), pool across snapshots, and fit
#            per-category [w_g, w_l, w_xgb] (non-negative, GTIN-aggregated) minimizing
#            WAPE + an asymmetric under-forecast penalty vs prediction_xgb. Guardrail: keep
#            the stack for a category only if it beats prediction_xgb on accuracy AND the
#            asymmetric-bias rule, else fall back to pure prediction_xgb.
#            → writes config/de_ensemble_weights.yaml (consumed unchanged by the LIVE run).
#
# Prerequisite: this repo must be on the cluster (Databricks Repos clone or workspace
# upload). Set REPO_PATH below, exactly like databricks_diq_notebook.py.
# =============================================================================

# COMMAND ----------

import os, sys
REPO_PATH = "/Workspace/Repos/<your_email>/FEUagenticForecastingUpgrade"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets — defaults are the confirmed DE parallel_run locations

# COMMAND ----------

dbutils.widgets.text("MARKET", "de")
dbutils.widgets.text("PARENT_DIR", "/Volumes/pds_feu_931272_dev/eu_dach/transform/DACH/DACH_LEGO/Germany/parallel_run")
dbutils.widgets.text("ALLCAT_SUBDIR", "All_Cat")            # {PARENT_DIR}/<YYYYWW>/{ALLCAT_SUBDIR}/...
dbutils.widgets.text("TRAIN_SNAPSHOTS", "")                 # blank → auto-discover (covered snapshots)
dbutils.widgets.text("LIVE_SNAPSHOT", "202621")            # excluded from the fit (it's the live target)
dbutils.widgets.text("ACTUALS_SNAPSHOT", "202625")         # DT supplies realized forward actuals
dbutils.widgets.text("PANEL_SUBDIR", "DT")                 # DIQ input panel (target=Actuals)
dbutils.widgets.text("FS_SUBDIR", "FS")                    # feature store (joined to DT)
dbutils.widgets.text("BENCHMARK_SUBDIR", "TF_pristine_allkeys")   # carries prediction_xgb
dbutils.widgets.text("BENCHMARK_COL", "prediction_xgb")
dbutils.widgets.text("MIN_FWD_WEEKS", "5")                 # min realized forward weeks to include a snapshot
dbutils.widgets.text("COMPONENTS_OUT", "/Volumes/pds_feu_931272_dev/eu_dach/transform/DACH/DACH_LEGO/Germany/parallel_run/_diq_fit/components")
dbutils.widgets.text("CONFIG_ROOT", "")                    # blank → <REPO_PATH>/config
dbutils.widgets.text("IN_SCOPE_CATEGORIES", "")            # blank → route ALL (guardrail handles weak cats)
dbutils.widgets.text("OBJECTIVE", "wape_asym")
dbutils.widgets.text("BIAS_TOLERANCE", "0.02")
dbutils.widgets.dropdown("GUARDRAIL", "true", ["true", "false"])
dbutils.widgets.dropdown("RUN_LEGO_IF_MISSING", "false", ["false", "true"])
dbutils.widgets.text("LEGO_RUN_NOTEBOOK", "")              # parametrized LEGO entry (for RUN_LEGO_IF_MISSING)
dbutils.widgets.dropdown("REUSE_SAVED_COMPONENTS", "false", ["false", "true"])  # skip Stage A, refit only
dbutils.widgets.text("HORIZONS", "13")                     # forward weeks the new DIQ fits per snapshot
# --- driver-memory controls for STAGE A (the new DIQ runs on the DRIVER) ---
# compute_components trains LightGBM across a ProcessPoolExecutor on the DRIVER; each worker holds a
# copy of the multi-category panel and burns ENGINE_THREADS cores. The engine default (~cores/threads
# − 1, i.e. ~7 on a 32-core driver) over-subscribes CPU + memory on a big multi-cat FIT → the driver
# stops answering health checks → DRIVER_NOT_RESPONDING / cluster crash. Start conservative (2×8 = 16
# of 32 cores, ~2 panel copies); raise toward 4 only if the driver has clear headroom.
dbutils.widgets.text("ENGINE_WORKERS", "2")                # ProcessPool workers (caps CPU + in-memory panel copies)
dbutils.widgets.text("ENGINE_THREADS", "8")                # LightGBM threads per worker

# COMMAND ----------

# MAGIC %md
# MAGIC ### Market presets — the SAME notebook fits either market; only widget values change
# MAGIC
# MAGIC **DE (defaults above):**
# MAGIC - `MARKET`=`de` · `PARENT_DIR`=`…/Germany/parallel_run` · `ALLCAT_SUBDIR`=`All_Cat`
# MAGIC - `PANEL_SUBDIR`=`DT` · `BENCHMARK_SUBDIR`=`TF_pristine_allkeys` · `BENCHMARK_COL`=`prediction_xgb`
# MAGIC - `ACTUALS_SNAPSHOT`=`202625` · `LIVE_SNAPSHOT`=`202621` · `TRAIN_SNAPSHOTS`=blank (auto-discover)
# MAGIC
# MAGIC **UK:**
# MAGIC - `MARKET`=`uk` · `PARENT_DIR`=`…/forward_run/total_forecast/snapshot` · `ALLCAT_SUBDIR`=**blank** (UK has no All_Cat level)
# MAGIC - `PANEL_SUBDIR`=`01_all_keys_dr_adjusted` · `BENCHMARK_SUBDIR`=`TF_modelling_pristine` · `BENCHMARK_COL`=`prediction_rf`
# MAGIC - `ACTUALS_SNAPSHOT`=`2026-25` · `LIVE_SNAPSHOT`=`2026-23`
# MAGIC - `TRAIN_SNAPSHOTS`=`2026-09#2026-10#2026-11#2026-12#2026-13#2026-14` (explicit — UK has 42 snapshots; don't auto-discover all)
# MAGIC - `COMPONENTS_OUT`=`…/forward_run/_diq_fit/components`

# COMMAND ----------

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format="%(asctime)s %(levelname).1s %(name)s | %(message)s")
from pyspark.sql import functions as F
from utils import weight_fit as wf

W = lambda k: dbutils.widgets.get(k).strip()
MARKET           = W("MARKET")
PARENT_DIR       = W("PARENT_DIR").rstrip("/")
ALLCAT           = W("ALLCAT_SUBDIR")
LIVE_SNAPSHOT    = W("LIVE_SNAPSHOT")
ACTUALS_SNAPSHOT = W("ACTUALS_SNAPSHOT")
PANEL_SUBDIR     = W("PANEL_SUBDIR")
FS_SUBDIR        = W("FS_SUBDIR")
BENCHMARK_SUBDIR = W("BENCHMARK_SUBDIR")
BENCHMARK_COL    = W("BENCHMARK_COL")
MIN_FWD_WEEKS    = int(W("MIN_FWD_WEEKS") or 5)
COMPONENTS_OUT   = W("COMPONENTS_OUT").rstrip("/")
CONFIG_ROOT      = W("CONFIG_ROOT") or os.path.join(REPO_PATH, "config")
IN_SCOPE         = [c.strip() for c in W("IN_SCOPE_CATEGORIES").replace(",", "#").split("#") if c.strip()] or None
OBJECTIVE        = W("OBJECTIVE") or "wape_asym"
BIAS_TOLERANCE   = float(W("BIAS_TOLERANCE") or 0.02)
GUARDRAIL        = W("GUARDRAIL").lower() == "true"
RUN_LEGO         = W("RUN_LEGO_IF_MISSING").lower() == "true"
LEGO_RUN_NOTEBOOK = W("LEGO_RUN_NOTEBOOK")
REUSE            = W("REUSE_SAVED_COMPONENTS").lower() == "true"
HORIZONS         = int(W("HORIZONS") or 13)

# Cap the new-DIQ ProcessPool fan-out so STAGE A doesn't exhaust the driver. The engine reads
# these env vars at compute_components time (utils/uk_engine.py): fewer workers → fewer in-memory
# copies of the multi-category panel. This is the fix for the DRIVER_NOT_RESPONDING crash.
os.environ["UK_ENGINE_WORKERS"] = W("ENGINE_WORKERS") or "3"
os.environ["UK_ENGINE_THREADS"] = W("ENGINE_THREADS") or "8"
print(f"ENGINE_WORKERS={os.environ['UK_ENGINE_WORKERS']} ENGINE_THREADS={os.environ['UK_ENGINE_THREADS']} HORIZONS={HORIZONS}")

# Actuals column name is market-specific — read it from the config (DE: Actuals, UK: actual_sales).
try:
    from utils.config_loader import load_market_config as _lmc
    ACTUALS_COL = _lmc(MARKET, CONFIG_ROOT).get("target", "Actuals")   # prefers .py, falls back to .yaml
except Exception:
    ACTUALS_COL = "Actuals"

def _join(base, *parts):
    """Path join that preserves the leading slash and tolerates an empty ALLCAT
    (DE nests under All_Cat/; UK has the subdirs directly under {snapshot}/)."""
    out = base.rstrip("/")
    for p in parts:
        if p:
            out += "/" + p.strip("/")
    return out

def allcat_dir(snap):  return _join(PARENT_DIR, snap, ALLCAT)
PARENT_DIR_TMPL = _join(PARENT_DIR, "{snapshot}", ALLCAT)
ACTUALS_SOURCE  = _join(allcat_dir(ACTUALS_SNAPSHOT), PANEL_SUBDIR)

print("MARKET:", MARKET, "| PARENT_DIR:", PARENT_DIR)
print("ACTUALS_SOURCE:", ACTUALS_SOURCE, "| LIVE (excluded):", LIVE_SNAPSHOT)
print("COMPONENTS_OUT:", COMPONENTS_OUT, "| CONFIG_ROOT:", CONFIG_ROOT)
print("IN_SCOPE:", IN_SCOPE or "ALL")

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE 0 — discover snapshots + check fit requirements

# COMMAND ----------

def _ls(path):
    try:
        return list(dbutils.fs.ls(path))
    except Exception:
        return []

def _has_parquet(path):
    return any(f.name.endswith(".parquet") or f.name.rstrip("/").endswith(".parquet") or f.name.startswith("part-")
               or f.isDir() for f in _ls(path))

def _yyyyww_int(s):
    s = str(s).replace("-", "")
    return int(s) if s.isdigit() and len(s) == 6 else None

def discover_snapshots():
    snaps = []
    for f in _ls(PARENT_DIR):
        name = f.name.rstrip("/")
        if _yyyyww_int(name) is not None:
            snaps.append(name)
    return sorted(set(snaps))

def lego_ready(snap):
    """All LEGO requirements present for this snapshot's All_Cat?"""
    pdir = allcat_dir(snap)
    need = {PANEL_SUBDIR: f"{pdir}/{PANEL_SUBDIR}",
            FS_SUBDIR: f"{pdir}/{FS_SUBDIR}",
            BENCHMARK_SUBDIR: f"{pdir}/{BENCHMARK_SUBDIR}"}
    missing = [k for k, p in need.items() if not _has_parquet(p)]
    if missing:
        return False, f"missing {missing}"
    # benchmark column populated?
    try:
        n = (spark.read.parquet(need[BENCHMARK_SUBDIR])
             .where(F.col(BENCHMARK_COL) > 0).limit(1).count())
        if n == 0:
            return False, f"{BENCHMARK_COL} all-zero/null"
    except Exception as e:
        return False, f"{BENCHMARK_COL} unreadable: {str(e)[:60]}"
    return True, "ready"

# realized-actuals horizon from the actuals snapshot
actuals_max_week = (spark.read.parquet(ACTUALS_SOURCE)
                    .where(F.col(ACTUALS_COL) > 0)
                    .agg(F.max(F.regexp_replace(F.col("year_week").cast("string"), "-", "")))
                    .first()[0])
actuals_max_int = int(actuals_max_week)
print("Actuals realized through:", actuals_max_week, f"(from {ACTUALS_SNAPSHOT})")

discovered = discover_snapshots()
print("Discovered snapshots:", discovered)

# build the readiness/coverage table. Keep snapshot ids in their DIR-NAME format
# (UK dirs are hyphenated '2026-09'; DE dirs are plain '202601') — they index the paths.
explicit = [s.strip() for s in W("TRAIN_SNAPSHOTS").replace(",", "#").split("#") if s.strip()]
candidates = explicit or [s for s in discovered if s not in (LIVE_SNAPSHOT, ACTUALS_SNAPSHOT)]

import pandas as pd
from utils.uk_forecast import add_weeks    # year-week arithmetic (handles year boundary)
rows, fit_snaps = [], []
for snap in candidates:
    # forward coverage: of weeks snap..snap+12, how many are realized (<= actuals_max)
    fwd_weeks = [add_weeks(snap, h) for h in range(0, HORIZONS)]
    covered = sum(1 for w in fwd_weeks if int(w) <= actuals_max_int)
    ready, why = lego_ready(snap)
    include = ready and covered >= MIN_FWD_WEEKS
    rows.append({"snapshot": snap, "lego_ready": ready, "lego_note": why,
                 "fwd_weeks_covered": covered, "include_in_fit": include})
    if include:
        fit_snaps.append(snap)

readiness = pd.DataFrame(rows)
try:
    display(readiness)                  # noqa: F821
except Exception:
    print(readiness.to_string(index=False))
print("\nFIT snapshots:", fit_snaps)
assert fit_snaps, "No snapshot satisfies the fit requirements (LEGO ready + enough forward actuals)."

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE A — per snapshot: skip LEGO if present (else run it), then run the new DIQ + SAVE

# COMMAND ----------

def ensure_lego(snap):
    """Make sure snapshot S has its LEGO artifacts (DT/, FS/, prediction_xgb) BEFORE the
    new DIQ runs — the DIQ consumes them. Reuse if present; else trigger the LEGO entry."""
    ready, why = lego_ready(snap)
    if ready:
        print(f"  [{snap}] LEGO present ({why}) → SKIP LEGO, run new DIQ only")
        return
    print(f"  [{snap}] LEGO NOT ready ({why})")
    if not RUN_LEGO:
        raise RuntimeError(
            f"Snapshot {snap} is missing LEGO ({why}). Generate it by running the LEGO entry "
            f"(DE: DE_AllCAT_PARALLEL_RUN with history_cut_off={add_weeks(snap, -1)}, which "
            f"%runs LEGO_RUN_SCRIPT_DE) for this snapshot, then re-run this notebook. Auto-run "
            f"is OFF — set RUN_LEGO_IF_MISSING=true + LEGO_RUN_NOTEBOOK (a PARAMETRIZED entry "
            f"that reads history_cut_off/snapshot_week/demandiq_run_flag widgets) to enable. "
            f"NOTE: regenerating a historical snapshot needs the point-in-time raw modelling "
            f"data for that week to exist upstream.")
    if not LEGO_RUN_NOTEBOOK:
        raise RuntimeError("RUN_LEGO_IF_MISSING=true but LEGO_RUN_NOTEBOOK is blank.")
    print(f"  [{snap}] triggering LEGO via {LEGO_RUN_NOTEBOOK} (DEMANDIQ_RUN_FLAG=false → skip old DIQ) …")
    # DEMANDIQ_RUN_FLAG=false → run DT→FE→FS→TF→Combine (DT/FS + prediction_xgb) but SKIP the
    # expensive old per-cat DIQ job fan-out; THIS notebook runs its own new DIQ next.
    # PARENT_DIR = the exact {snap} All_Cat dir we read from, so LEGO lands precisely there.
    # Widget names match LEGO_RUN_DE_PARAM (the parametrized DE LEGO entry).
    dbutils.notebook.run(LEGO_RUN_NOTEBOOK, 6 * 60 * 60, {
        "HISTORY_CUT_OFF": add_weeks(snap, -1), "SNAPSHOT_WEEK": snap,
        "DEMANDIQ_RUN_FLAG": "false", "PARENT_DIR": allcat_dir(snap), "MARKET": MARKET})
    ready, why = lego_ready(snap)
    if not ready:
        raise RuntimeError(f"LEGO run for {snap} did not satisfy requirements: {why}")

saved_paths = []
if REUSE:
    print("REUSE_SAVED_COMPONENTS=true → skipping Stage A.")
    saved_paths = [f"{COMPONENTS_OUT}/components_{s}.parquet" for s in fit_snaps]
else:
    for snap in fit_snaps:
        ensure_lego(snap)
        info = wf.save_components(
            MARKET, snap,
            out_dir=COMPONENTS_OUT,
            config_root=CONFIG_ROOT,
            parent_dir_template=PARENT_DIR_TMPL,     # DT⋈FS panel, like LIVE
            lego_path_template=_join(PARENT_DIR_TMPL, BENCHMARK_SUBDIR),
            in_scope_categories=IN_SCOPE,
            horizons=HORIZONS,
        )
        saved_paths.append(info["path"])
        print(f"  [A] {snap}: {info['rows']:>8,} rows, {len(info['categories'])} cats -> {info['path']}")
    print("\nStage A complete:", len(saved_paths), "snapshots' components saved.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE B — combine all DIQ+LEGO runs and fit the stacked weights

# COMMAND ----------

result = wf.fit_from_components(
    MARKET,
    component_paths=saved_paths,             # only the fit-set snapshots (no stale parquet)
    actuals_source=ACTUALS_SOURCE,
    config_root=CONFIG_ROOT,
    objective=OBJECTIVE,
    bias_tolerance=BIAS_TOLERANCE,
    guardrail=GUARDRAIL,
    return_detail=True,
)
weights_path, fitted = result["path"], result["fitted"]
print("Weights written to:", weights_path)

# Volume copy so the weights survive the cluster.
import shutil
try:
    os.makedirs(COMPONENTS_OUT, exist_ok=True)
    shutil.copyfile(weights_path, f"{COMPONENTS_OUT}/{MARKET}_ensemble_weights.yaml")
    print("Volume copy        :", f"{COMPONENTS_OUT}/{MARKET}_ensemble_weights.yaml")
except Exception as e:
    print("Volume copy skipped:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoreboard — fitted weights + in-sample acc/bias (stack vs prediction_xgb)

# COMMAND ----------

rows = []
for cat, v in sorted(fitted.items()):
    wg, wl, wx = v["weights"]
    rows.append({"category": cat, "use": v["use"], "w_global": wg, "w_local": wl, "w_xgb": wx,
                 "acc_xgb": v["acc_lego"], "acc_stack": v["acc_stack"],
                 "acc_delta": round(v["acc_stack"] - v["acc_lego"], 4),
                 "bias_xgb": v["bias_lego"], "bias_stack": v["bias_stack"]})
score = pd.DataFrame(rows)
print(f"STACK deployed for {int((score['use']=='STACK').sum())}/{len(score)} categories "
      f"(rest fall back to pure prediction_xgb).")
try:
    display(score)                          # noqa: F821
except Exception:
    print(score.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next step — LIVE run
# MAGIC
# MAGIC Commit the written `config/de_ensemble_weights.yaml`, then run the LIVE forward
# MAGIC forecast on `LIVE_SNAPSHOT` (202621), which applies these frozen weights unchanged:
# MAGIC ```
# MAGIC python run_stack.py --mode live --market de
# MAGIC ```
# MAGIC (or the inline `run_diq_forecast(..., market="de", enable_uk_engine=True, ...)` cell),
# MAGIC then compare 202621 forecasts vs actuals on accuracy + the asymmetric bias rule.
# MAGIC Re-run THIS notebook anytime to refresh the weights.
