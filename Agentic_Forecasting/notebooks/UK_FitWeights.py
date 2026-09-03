# Databricks notebook source
# =============================================================================
# UK_FitWeights — STEP 1 of 2: calculate the ensemble weights (UK)
# =============================================================================
# Identical methodology to DE_FitWeights — the segment×category×LEGO stacked ensemble —
# only the market defaults differ (resolved from config_uk_base_local.py). For each
# TRAINING snapshot S:
#   * LEGO present  (01_all_keys_dr_adjusted/, FS/, TF_pristine_allkeys.prediction_rf) -> SKIP LEGO
#   * LEGO missing  + RUN_LEGO_IF_MISSING=true -> build it via dbutils.notebook.run on the UK
#                     forward-run LEGO entry (set LEGO_RUN_NOTEBOOK). Default OFF for UK (the
#                     forward_run snapshots already carry LEGO; produce missing ones upstream).
#   * run the NEW DIQ (segment components) and save them
# Then pool every snapshot + realized actuals and fit the per-(category x lego_segment)
# weights -> config/uk_ensemble_weights.yaml  (+ a Volume copy that survives the cluster).
#
# Calls the SAME validated functions as DE (utils.weight_fit) — a turnkey UK preset, NOT a
# reimplementation. Set REPO_PATH (or rely on auto-detect) and "Run All".
#
# STEP 2 = UK_Forecast: takes the weights this writes and produces the forward forecast.
# =============================================================================

# COMMAND ----------

import os, sys
REPO_PATH = ""
try:
    _nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    _root = os.path.dirname(os.path.dirname(_nb))            # …/notebooks/UK_FitWeights -> repo root
    if os.path.isdir(f"/Workspace{_root}/utils"):
        REPO_PATH = f"/Workspace{_root}"
except Exception:
    pass
if not REPO_PATH or not os.path.isdir(os.path.join(REPO_PATH, "utils")):
    REPO_PATH = "/Workspace/Repos/<your_email>/Agentic_Forecasting"   # <-- fallback: set me
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
print("REPO_PATH:", REPO_PATH)

# COMMAND ----------

# ---- UK config — defaults are the confirmed forward_run/total_forecast/snapshot locations ----
dbutils.widgets.text("PARENT_DIR",       "/Volumes/pds_feu_931272_dev/eu_uk/landing/UK_PROD_NEW/lego_runs/forward_run/total_forecast/snapshot")
dbutils.widgets.text("TRAIN_SNAPSHOTS",  "2026-09,2026-10,2026-11,2026-12,2026-13,2026-14")  # blank -> auto-discover under PARENT_DIR
dbutils.widgets.text("LIVE_SNAPSHOT",    "2026-23")                      # EXCLUDED from the fit (it's the live target)
dbutils.widgets.text("ACTUALS_SNAPSHOT", "2026-25")                      # realized forward actuals (actual_sales)
dbutils.widgets.dropdown("RUN_LEGO_IF_MISSING", "false", ["true", "false"])  # UK LEGO produced upstream; OFF by default
dbutils.widgets.text("LEGO_RUN_NOTEBOOK", "")            # UK forward-run LEGO entry (set if RUN_LEGO_IF_MISSING=true)
dbutils.widgets.text("ENGINE_WORKERS",   "6")            # ProcessPool workers on the DRIVER (each ≈ 1 panel copy in RAM)
dbutils.widgets.text("ENGINE_THREADS",   "4")            # LightGBM threads / worker  (workers × threads ≈ driver cores)
dbutils.widgets.text("HORIZONS",         "13")           # forward weeks the DIQ fits per snapshot
dbutils.widgets.text("MIN_FWD_WEEKS",    "5")            # min realized forward weeks to include a snapshot
dbutils.widgets.text("OBJECTIVE",        "wape_asym")
dbutils.widgets.text("BIAS_TOLERANCE",   "0.02")
# --- UK guardrail ON by default (UK LEGO is a stronger ceiling — keep the robust fallback) ---
dbutils.widgets.dropdown("GUARDRAIL",    "true", ["true", "false"])
dbutils.widgets.text("SEG_SHRINK_K",     "5000")         # prod default shrink toward category (0 = raw per-segment)
dbutils.widgets.text("SEG_MIN_VOLUME",   "1000")         # prod default low-volume floor -> [0,0,1] (0 = fit all)

W = lambda k: dbutils.widgets.get(k).strip()
PARENT_DIR        = W("PARENT_DIR").rstrip("/")
LIVE_SNAPSHOT     = W("LIVE_SNAPSHOT")
ACTUALS_SNAPSHOT  = W("ACTUALS_SNAPSHOT")
RUN_LEGO          = W("RUN_LEGO_IF_MISSING").lower() == "true"
LEGO_RUN_NOTEBOOK = W("LEGO_RUN_NOTEBOOK")
HORIZONS          = int(W("HORIZONS"))
MIN_FWD_WEEKS     = int(W("MIN_FWD_WEEKS"))
OBJECTIVE         = W("OBJECTIVE")
BIAS_TOLERANCE    = float(W("BIAS_TOLERANCE"))
GUARDRAIL         = W("GUARDRAIL").lower() == "true"
os.environ["UK_ENGINE_WORKERS"] = W("ENGINE_WORKERS")    # the segment engine reads these
os.environ["UK_ENGINE_THREADS"] = W("ENGINE_THREADS")
os.environ["SEG_SHRINK_K"]      = W("SEG_SHRINK_K")
os.environ["SEG_MIN_VOLUME"]    = W("SEG_MIN_VOLUME")

# UK conventions (mirror config_uk_base_local.py market_io). NOTE: UK has NO All_Cat level —
# the snapshot dir is {PARENT_DIR}/{snap} directly, panel under it at 01_all_keys_dr_adjusted.
MARKET, PANEL, FS_SUB = "uk", "01_all_keys_dr_adjusted", "FS"
BENCH_SUB, BENCH_COL, ACTUALS_COL = "TF_pristine_allkeys", "prediction_rf", "actual_sales"
CONFIG_ROOT     = os.path.join(REPO_PATH, "config")
COMPONENTS_OUT  = f"{os.path.dirname(PARENT_DIR)}/_diq_fit/components"   # …/forward_run/_diq_fit/components
snap_dir        = lambda s: f"{PARENT_DIR}/{s}"                          # no All_Cat for UK
PARENT_DIR_TMPL = f"{PARENT_DIR}/{{snapshot}}"
ACTUALS_SOURCE  = f"{snap_dir(ACTUALS_SNAPSHOT)}/{PANEL}"

import pyspark.sql.functions as F
import pandas as pd
from utils import weight_fit as wf
from utils.uk_forecast import add_weeks          # year-week arithmetic (handles year boundary)
print("MARKET:", MARKET, "| PARENT_DIR:", PARENT_DIR, "| RUN_LEGO_IF_MISSING:", RUN_LEGO)
print("CONFIG_ROOT:", CONFIG_ROOT, "| COMPONENTS_OUT:", COMPONENTS_OUT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE 0 — readiness: which training snapshots have LEGO + enough realized actuals

# COMMAND ----------

def _ls(p):
    try: return list(dbutils.fs.ls(p))
    except Exception: return []

def _has_parquet(p):
    return any(f.name.endswith(".parquet") or f.name.startswith("part-") or f.isDir() for f in _ls(p))

def lego_ready(snap):
    """All LEGO requirements present for this snapshot (01_all_keys_dr_adjusted/, FS/, prediction_rf)?"""
    pdir = snap_dir(snap)
    need = {PANEL: f"{pdir}/{PANEL}", FS_SUB: f"{pdir}/{FS_SUB}", BENCH_SUB: f"{pdir}/{BENCH_SUB}"}
    missing = [k for k, p in need.items() if not _has_parquet(p)]
    if missing:
        return False, f"missing {missing}"
    try:
        if spark.read.parquet(need[BENCH_SUB]).where(F.col(BENCH_COL) > 0).limit(1).count() == 0:
            return False, f"{BENCH_COL} all-zero/null"
    except Exception as e:
        return False, f"{BENCH_COL} unreadable: {str(e)[:50]}"
    return True, "ready"

def discover():
    out = []
    for f in _ls(PARENT_DIR):
        n = f.name.rstrip("/"); d = n.replace("-", "")
        if d.isdigit() and len(d) == 6:
            out.append(n)
    return sorted(set(out))

actuals_max = int(str(
    spark.read.parquet(ACTUALS_SOURCE).where(F.col(ACTUALS_COL) > 0)
    .agg(F.max(F.regexp_replace(F.col("year_week").cast("string"), "-", ""))).first()[0]))
print("Actuals realized through:", actuals_max, f"(from {ACTUALS_SNAPSHOT})")

explicit   = [s.strip() for s in W("TRAIN_SNAPSHOTS").replace(",", "#").split("#") if s.strip()]
candidates = explicit or [s for s in discover() if s not in (LIVE_SNAPSHOT, ACTUALS_SNAPSHOT)]
rows, fit_snaps = [], []
for snap in candidates:
    covered    = sum(1 for h in range(HORIZONS) if int(add_weeks(snap, h)) <= actuals_max)
    ready, why = lego_ready(snap)
    include    = ready and covered >= MIN_FWD_WEEKS
    rows.append({"snapshot": snap, "lego_ready": ready, "note": why,
                 "fwd_weeks_covered": covered, "include_in_fit": include})
    if include:
        fit_snaps.append(snap)

try:    display(pd.DataFrame(rows))                                   # noqa: F821
except Exception: print(pd.DataFrame(rows).to_string(index=False))
print("\nFIT snapshots:", fit_snaps)
assert fit_snaps, "No snapshot is fit-ready (need LEGO present + >= MIN_FWD_WEEKS realized actuals)."

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE A — ensure LEGO (skip if present, else run it) + save the DIQ components

# COMMAND ----------

def ensure_lego(snap):
    ready, why = lego_ready(snap)
    if ready:
        print(f"  [{snap}] LEGO present -> SKIP LEGO, run DIQ only")
        return
    if not RUN_LEGO:
        raise RuntimeError(
            f"[{snap}] LEGO missing ({why}) and RUN_LEGO_IF_MISSING=false. Produce the UK forward-run "
            f"LEGO for {snap} upstream first, or set RUN_LEGO_IF_MISSING=true + LEGO_RUN_NOTEBOOK.")
    if not LEGO_RUN_NOTEBOOK:
        raise RuntimeError("RUN_LEGO_IF_MISSING=true but LEGO_RUN_NOTEBOOK is blank (set the UK LEGO entry).")
    print(f"  [{snap}] LEGO missing ({why}) -> running {LEGO_RUN_NOTEBOOK} ...")
    dbutils.notebook.run(LEGO_RUN_NOTEBOOK, 6 * 60 * 60, {
        "SNAPSHOT_WEEK": snap, "PARENT_DIR": snap_dir(snap), "MARKET": MARKET})
    ready, why = lego_ready(snap)
    if not ready:
        raise RuntimeError(f"[{snap}] LEGO run did not satisfy requirements: {why}")

saved = []
for snap in fit_snaps:
    ensure_lego(snap)
    info = wf.save_components(
        MARKET, snap,
        out_dir=COMPONENTS_OUT,
        config_root=CONFIG_ROOT,
        parent_dir_template=PARENT_DIR_TMPL,                  # panel join FS, exactly like LIVE
        lego_path_template=f"{PARENT_DIR_TMPL}/{BENCH_SUB}",  # prediction_rf benchmark
        in_scope_categories=None,                             # route all; guardrail handles weak cats
        horizons=HORIZONS,
    )
    saved.append(info["path"])
    print(f"  [A] {snap}: {info['rows']:>8,} rows, {len(info['categories'])} cats -> {info['path']}")
print("\nStage A complete:", len(saved), "snapshots' components saved.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE B — pool components + actuals, fit the per-(category x segment) weights

# COMMAND ----------

result = wf.fit_from_components(
    MARKET,
    component_paths=saved,
    actuals_source=ACTUALS_SOURCE,
    config_root=CONFIG_ROOT,
    objective=OBJECTIVE,
    bias_tolerance=BIAS_TOLERANCE,
    guardrail=GUARDRAIL,
    return_detail=True,
)
weights_path, fitted = result["path"], result["fitted"]
print("Weights written to:", weights_path)

import shutil
VOLUME_WEIGHTS = f"{COMPONENTS_OUT}/{MARKET}_ensemble_weights.yaml"
try:
    os.makedirs(COMPONENTS_OUT, exist_ok=True)
    shutil.copyfile(weights_path, VOLUME_WEIGHTS)
    print("Volume copy        :", VOLUME_WEIGHTS)
except Exception as e:
    print("Volume copy skipped:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoreboard — fitted weights + in-sample acc/bias (stack vs prediction_rf)

# COMMAND ----------

sb = []
for key, v in sorted(fitted.items(), key=lambda kv: str(kv[0])):
    try:
        wg, wl, wx = v["weights"]
        sb.append({"group": str(key), "use": v.get("use"),
                   "w_seg": wg, "w_cat": wl, "w_rf": wx,
                   "acc_rf": v.get("acc_lego"), "acc_stack": v.get("acc_stack"),
                   "acc_delta": round(v.get("acc_stack", 0) - v.get("acc_lego", 0), 4),
                   "bias_rf": v.get("bias_lego"), "bias_stack": v.get("bias_stack")})
    except Exception:
        sb.append({"group": str(key), "use": str(v)[:40]})
score = pd.DataFrame(sb)
n_stack = int((score.get("use") == "STACK").sum()) if "use" in score else 0
print(f"STACK deployed for {n_stack}/{len(score)} (category x segment) groups (rest fall back to pure prediction_rf).")
try:    display(score)                                                # noqa: F821
except Exception: print(score.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done — weights fitted.  NEXT: STEP 2 = `UK_Forecast`
# MAGIC
# MAGIC `config/uk_ensemble_weights.yaml` is written (and copied to the Volume,
# MAGIC printed above as `VOLUME_WEIGHTS`). Run **`UK_Forecast`** next to apply these
# MAGIC weights and produce the forward forecast.
