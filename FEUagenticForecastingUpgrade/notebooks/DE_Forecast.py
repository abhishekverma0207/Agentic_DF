# Databricks notebook source
# =============================================================================
# DE_Forecast — STEP 2 of 2: apply the fitted weights -> forward forecast
# =============================================================================
# Uses the weights from STEP 1 (DE_FitWeights -> config/de_ensemble_weights.yaml).
# For the LIVE snapshot:
#   * LEGO present  (DT/, FS/, TF_pristine_allkeys.prediction_xgb) -> SKIP LEGO
#   * LEGO missing  + RUN_LEGO_IF_MISSING=true -> build it via dbutils.notebook.run on
#                     LEGO_RUN_DE_PARAM (DEMANDIQ_RUN_FLAG=false: LEGO + prediction_xgb only)
#   * run the DIQ (segment engine) applying the FROZEN weights -> write the forecast parquet
#
# Same call the production fan-out makes (DIQ_run_wrapper -> run_diq_forecast, market=de),
# but in ONE notebook over all categories, and it ENSURES the LEGO benchmark exists BEFORE
# forecasting (so there's no read-before-write ordering gotcha). Set REPO_PATH, "Run All".
# =============================================================================

# COMMAND ----------

import os, sys
REPO_PATH = ""   # auto-detect this notebook's repo root; else hard-set below
try:
    _nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    _root = os.path.dirname(os.path.dirname(_nb))
    if os.path.isdir(f"/Workspace{_root}/utils"):
        REPO_PATH = f"/Workspace{_root}"
except Exception:
    pass
if not REPO_PATH or not os.path.isdir(os.path.join(REPO_PATH, "utils")):
    REPO_PATH = "/Workspace/Repos/<your_email>/FEUagenticForecastingUpgrade"   # <-- fallback: set me
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
print("REPO_PATH:", REPO_PATH)

# COMMAND ----------

# ---- DE LIVE config ----
dbutils.widgets.text("PARENT_DIR",     "/Volumes/pds_feu_931272_dev/eu_dach/transform/DACH/DACH_LEGO/Germany/parallel_run")
dbutils.widgets.text("LIVE_SNAPSHOT",  "202621")     # week to forecast FROM (first forward week)
dbutils.widgets.text("HISTORY_TILL",   "202620")     # last week of known actuals (= LIVE_SNAPSHOT - 1)
dbutils.widgets.text("CATEGORY_LIST",
    "SCRATCH COOKING AIDS#DEODORANTS & FRAGRANCES#DRESSINGS#SKIN CLEANSING#HEALTHY SNACKING#"
    "HOME & HYGIENE#FABRIC CLEANING#ORAL CARE#SKIN CARE#HAIR CARE")              # '#'- or ','-separated
dbutils.widgets.text("WEIGHTS_PATH",   "")           # blank -> config/de_ensemble_weights.yaml (committed)
dbutils.widgets.text("OUT_PATH",       "")           # blank -> {LIVE All_Cat}/DIQ/TF_DIQ/de_{snap}_inference_forecast.parquet
dbutils.widgets.dropdown("RUN_LEGO_IF_MISSING", "true", ["true", "false"])
dbutils.widgets.text("LEGO_RUN_NOTEBOOK",
    "/Workspace/Users/anu.thomas@unilever.com/DACH_Industrialized/ParallelRun_Scripts/DE/LEGO_RUN_DE_PARAM")
dbutils.widgets.text("ENGINE_WORKERS", "2")
dbutils.widgets.text("ENGINE_THREADS", "8")
dbutils.widgets.dropdown("UPDATE_TF_PRISTINE", "true", ["true", "false"])  # write ml_pred_final into TF_pristine_allkeys
dbutils.widgets.text("LEGO_PRED_COL", "prediction_xgb")  # ml_pred_final fallback where a key isn't in the DIQ forecast
dbutils.widgets.text("TF_PRISTINE_PATH", "")             # blank -> {LIVE All_Cat}/TF_pristine_allkeys

W = lambda k: dbutils.widgets.get(k).strip()
PARENT_DIR        = W("PARENT_DIR").rstrip("/")
LIVE_SNAPSHOT     = W("LIVE_SNAPSHOT")
HISTORY_TILL      = W("HISTORY_TILL")
RUN_LEGO          = W("RUN_LEGO_IF_MISSING").lower() == "true"
LEGO_RUN_NOTEBOOK = W("LEGO_RUN_NOTEBOOK")
os.environ["UK_ENGINE_WORKERS"] = W("ENGINE_WORKERS")
os.environ["UK_ENGINE_THREADS"] = W("ENGINE_THREADS")

MARKET, ALLCAT, PANEL, FS_SUB = "de", "All_Cat", "DT", "FS"
BENCH_SUB, BENCH_COL = "TF_pristine_allkeys", "prediction_xgb"
CONFIG_ROOT = os.path.join(REPO_PATH, "config")
allcat_dir  = lambda s: f"{PARENT_DIR}/{s}/{ALLCAT}"
LIVE_DIR    = allcat_dir(LIVE_SNAPSHOT)

cats         = [c.strip() for c in W("CATEGORY_LIST").replace(",", "#").split("#") if c.strip()]
WEIGHTS_PATH = W("WEIGHTS_PATH") or f"{CONFIG_ROOT}/de_ensemble_weights.yaml"
OUT_PATH     = W("OUT_PATH") or f"{LIVE_DIR}/DIQ/TF_DIQ/de_{LIVE_SNAPSHOT}_inference_forecast.parquet"
UPDATE_TF     = W("UPDATE_TF_PRISTINE").lower() == "true"
LEGO_PRED_COL = W("LEGO_PRED_COL") or "prediction_xgb"
TF_PRISTINE   = W("TF_PRISTINE_PATH") or f"{LIVE_DIR}/{BENCH_SUB}"   # BENCH_SUB = TF_pristine_allkeys

import pyspark.sql.functions as F
import pandas as pd
from utils.uk_forecast import add_weeks
print("LIVE snapshot:", LIVE_SNAPSHOT, "| history_till:", HISTORY_TILL, "| categories:", len(cats))
print("WEIGHTS_PATH :", WEIGHTS_PATH)
print("OUT_PATH     :", OUT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE A — ensure the LEGO benchmark for the LIVE snapshot exists (skip if present, else run it)

# COMMAND ----------

def _ls(p):
    try: return list(dbutils.fs.ls(p))
    except Exception: return []

def _has_parquet(p):
    return any(f.name.endswith(".parquet") or f.name.startswith("part-") or f.isDir() for f in _ls(p))

def lego_ready(snap):
    pdir = allcat_dir(snap)
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

ready, why = lego_ready(LIVE_SNAPSHOT)
if ready:
    print(f"[{LIVE_SNAPSHOT}] LEGO present -> SKIP LEGO")
elif not RUN_LEGO:
    raise RuntimeError(f"[{LIVE_SNAPSHOT}] LEGO missing ({why}) and RUN_LEGO_IF_MISSING=false. Run "
                       f"DE_AllCAT_PARALLEL_RUN for {LIVE_SNAPSHOT} first, or set RUN_LEGO_IF_MISSING=true.")
else:
    if not LEGO_RUN_NOTEBOOK:
        raise RuntimeError("RUN_LEGO_IF_MISSING=true but LEGO_RUN_NOTEBOOK is blank.")
    print(f"[{LIVE_SNAPSHOT}] LEGO missing ({why}) -> running {LEGO_RUN_NOTEBOOK} (DEMANDIQ_RUN_FLAG=false) ...")
    dbutils.notebook.run(LEGO_RUN_NOTEBOOK, 6 * 60 * 60, {
        "HISTORY_CUT_OFF": add_weeks(LIVE_SNAPSHOT, -1), "SNAPSHOT_WEEK": LIVE_SNAPSHOT,
        "DEMANDIQ_RUN_FLAG": "false", "PARENT_DIR": LIVE_DIR, "MARKET": MARKET})
    ready, why = lego_ready(LIVE_SNAPSHOT)
    if not ready:
        raise RuntimeError(f"[{LIVE_SNAPSHOT}] LEGO run did not produce required artifacts: {why}")

# weights from STEP 1 must exist
if not os.path.exists(WEIGHTS_PATH):
    raise RuntimeError(f"Weights not found at {WEIGHTS_PATH}. Run STEP 1 (DE_FitWeights) first, or point "
                       f"WEIGHTS_PATH at the committed config or the Volume copy.")
print("Weights found:", WEIGHTS_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE B — run the DIQ applying the FROZEN weights -> write the forward forecast

# COMMAND ----------

from utils.diq_runner import run_diq_forecast
stacked = run_diq_forecast(
    parent_dir=LIVE_DIR,
    history_till=HISTORY_TILL,
    snapshot_week=LIVE_SNAPSHOT,
    category_list=cats,
    out_path=OUT_PATH,
    market=MARKET,
    cats_subdir=PANEL,            # DT
    fs_subdir=FS_SUB,             # FS
    category_col="Category",
    config_root=CONFIG_ROOT,
    uk_weights_path=WEIGHTS_PATH, # the fitted per-(category x segment) weights from STEP 1
)
print(f"Forecast rows: {len(stacked):,} | categories: {stacked['category'].nunique()}")
print(f"Written to   : {OUT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE C — write the DIQ stacked forecast into `TF_pristine_allkeys.ml_pred_final`
# MAGIC LEFT-join onto TF_pristine_allkeys so **every row is kept**. `ml_pred_final` = the DIQ stacked
# MAGIC `predicted` where the (key, year_week) is in the DIQ forecast, else the LEGO column
# MAGIC (`LEGO_PRED_COL`, default `prediction_xgb`). A hard row-count assertion + temp-swap overwrite
# MAGIC guarantee no rows are lost. Set `UPDATE_TF_PRISTINE=false` to skip.

# COMMAND ----------

if not UPDATE_TF:
    print("UPDATE_TF_PRISTINE=false -> skipping the TF_pristine_allkeys update.")
else:
    # DIQ stacked output -> normalized + de-duped on (key, year_week); unique helper names avoid join collisions
    diq = stacked[["key", "year_week", "predicted"]].copy()
    diq["_dk"] = diq["key"].astype(str)
    diq["_dyw"] = diq["year_week"].astype(str).str.replace("-", "", regex=False)
    diq = (diq.groupby(["_dk", "_dyw"], as_index=False)["predicted"].sum()
              .rename(columns={"predicted": "_diq_pred"}))
    diq_sdf = spark.createDataFrame(diq[["_dk", "_dyw", "_diq_pred"]])

    tf = spark.read.parquet(TF_PRISTINE)
    n0 = tf.count()
    cols0 = tf.columns
    if "ml_pred_final" not in cols0:
        raise RuntimeError(f"{TF_PRISTINE} has no ml_pred_final column; columns = {cols0}")
    fallback = LEGO_PRED_COL if LEGO_PRED_COL in cols0 else "ml_pred_final"
    print(f"TF_pristine rows={n0:,} | DIQ (key×week)={len(diq):,} | unmatched fallback col = '{fallback}'")

    # LEFT join (keep every TF_pristine row); normalize year_week on both sides to match formats
    tfj = (tf.withColumn("_k", F.col("key").cast("string"))
             .withColumn("_yw", F.regexp_replace(F.col("year_week").cast("string"), "-", "")))
    j = tfj.join(diq_sdf, (tfj["_k"] == diq_sdf["_dk"]) & (tfj["_yw"] == diq_sdf["_dyw"]), "left")
    # ml_pred_final = DIQ stacked where matched, else the LEGO column (else keep the existing value)
    j = j.withColumn("ml_pred_final", F.coalesce(F.col("_diq_pred"), F.col(fallback), F.col("ml_pred_final")))
    if "source" in cols0:
        j = j.withColumn("source", F.when(F.col("_diq_pred").isNotNull(),
                                           F.lit("ML_FCST+DIQ_STACK")).otherwise(F.col("source")))
    n_diq = j.where(F.col("_diq_pred").isNotNull()).count()
    result = j.select(cols0)                         # back to EXACT original schema/order (drops helpers)
    n1 = result.count()
    assert n1 == n0, f"ROW COUNT CHANGED {n0} -> {n1} — aborting, NOT writing TF_pristine_allkeys."

    # safe in-place overwrite: write temp -> re-read temp -> overwrite original FROM temp -> drop temp
    tmp = TF_PRISTINE.rstrip("/") + "_diqtmp"
    result.write.mode("overwrite").parquet(tmp)
    staged = spark.read.parquet(tmp)
    assert staged.count() == n0, "temp row-count mismatch — aborting before overwrite."
    staged.write.mode("overwrite").parquet(TF_PRISTINE)
    dbutils.fs.rm(tmp, recurse=True)
    print(f"✅ {TF_PRISTINE}: {n0:,} rows kept | ml_pred_final = DIQ on {n_diq:,} rows, "
          f"LEGO('{fallback}') on {n0 - n_diq:,}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Peek — the forward forecast (`predicted` = the segment x category x LEGO blend)

# COMMAND ----------

try:
    display(spark.read.parquet(OUT_PATH).orderBy("category", "key", "year_week").limit(50))   # noqa: F821
except Exception as e:
    print("spark preview skipped:", e)
    print(stacked.head(20).to_string(index=False))
