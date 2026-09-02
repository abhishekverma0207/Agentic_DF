# Databricks notebook source
# =============================================================================
# FR_Forecast — STEP 2 of 2: apply the fitted weights -> forward forecast
# =============================================================================
# Uses the weights from STEP 1 (FR_FitWeights -> config/fr_ensemble_weights.yaml).
# For the LIVE snapshot:
#   * LEGO must already be present (DT/, FS/, TF with prediction_rf)
#   * run the DIQ (segment engine) applying the FROZEN weights -> write the forecast parquet
#
# Same structure as DE_Forecast but with FR paths and RUN_LEGO_IF_MISSING=false.
# Uses run_diq_forecast (DT⋈FS join) since France has the same folder structure as DE.
#
# STEP 1 = FR_FitWeights (must be run first to produce the weights YAML).
# =============================================================================

# COMMAND ----------

import os, sys
REPO_PATH = ""
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

# ---- FR LIVE config ----
dbutils.widgets.text("PARENT_DIR",     "/Volumes/pds_feu_931272_dev/eu_france/transform/FR/DIQ_Test/lego_output")
dbutils.widgets.text("LIVE_SNAPSHOT",  "202620")         # week to forecast FROM (first forward week)
dbutils.widgets.text("HISTORY_TILL",   "202619")         # last week of known actuals (= LIVE_SNAPSHOT - 1)
dbutils.widgets.text("CATEGORY_LIST",
    "CONDIMENT#COOKING AID#DEODORANT & FRAGRANCE#FABRIC CLEANING#FABRIC ENHANCER#"
    "HAIR CARE#HOME & HYGIENE#ICE CREAM CATEGORY#MINI MEAL#ORAL CARE#"
    "OTH FOOD#PLANT BASED MEAT#SKIN CLEANSING")
dbutils.widgets.text("WEIGHTS_PATH",   "")               # blank -> config/fr_ensemble_weights.yaml (committed)
dbutils.widgets.text("OUT_PATH",       "")               # blank -> {LIVE All_Cat}/DIQ/TF_DIQ/fr_{snap}_inference_forecast.parquet
dbutils.widgets.text("ENGINE_WORKERS", "2")
dbutils.widgets.text("ENGINE_THREADS", "8")
dbutils.widgets.dropdown("UPDATE_TF_PRISTINE", "false", ["true", "false"])
dbutils.widgets.text("LEGO_PRED_COL", "prediction_rf")
dbutils.widgets.text("TF_PRISTINE_PATH", "")             # blank -> {LIVE All_Cat}/TF

W = lambda k: dbutils.widgets.get(k).strip()
PARENT_DIR        = W("PARENT_DIR").rstrip("/")
LIVE_SNAPSHOT     = W("LIVE_SNAPSHOT")
HISTORY_TILL      = W("HISTORY_TILL")
os.environ["UK_ENGINE_WORKERS"] = W("ENGINE_WORKERS")
os.environ["UK_ENGINE_THREADS"] = W("ENGINE_THREADS")

MARKET, PANEL, FS_SUB = "fr", "DT", "FS"
BENCH_SUB, BENCH_COL = "TF", "prediction_rf"
CONFIG_ROOT = os.path.join(REPO_PATH, "config")
allcat_dir  = lambda s: f"{PARENT_DIR}/{s}"
LIVE_DIR    = allcat_dir(LIVE_SNAPSHOT)

cats         = [c.strip() for c in W("CATEGORY_LIST").replace(",", "#").split("#") if c.strip()]
WEIGHTS_PATH = W("WEIGHTS_PATH") or f"{CONFIG_ROOT}/fr_ensemble_weights.yaml"
OUT_PATH     = W("OUT_PATH") or f"{LIVE_DIR}/DIQ/TF_DIQ/fr_{LIVE_SNAPSHOT}_inference_forecast.parquet"
UPDATE_TF     = W("UPDATE_TF_PRISTINE").lower() == "true"
LEGO_PRED_COL = W("LEGO_PRED_COL") or "prediction_rf"
TF_PRISTINE   = W("TF_PRISTINE_PATH") or f"{LIVE_DIR}/{BENCH_SUB}"

os.environ["DIQ_RUNNER_SKIP_INSTALL"] = "1"

import pyspark.sql.functions as F
import pandas as pd
from utils.uk_forecast import add_weeks
print("LIVE snapshot:", LIVE_SNAPSHOT, "| history_till:", HISTORY_TILL, "| categories:", len(cats))
print("WEIGHTS_PATH :", WEIGHTS_PATH)
print("OUT_PATH     :", OUT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE A — verify LEGO for the LIVE snapshot + weights exist

# COMMAND ----------

def _ls(p):
    try: return list(dbutils.fs.ls(p))
    except Exception: return []

def _exists(p):
    try:
        dbutils.fs.ls(p)
        return True
    except Exception:
        return False

def _fmt(p):
    """Delta if the path carries a _delta_log transaction log, else parquet."""
    return "delta" if _exists(f"{p}/_delta_log") else "parquet"

def read_any(p):
    """Read a path without caring whether it was written as Delta or parquet."""
    fmt = _fmt(p)
    print(f"  [read_any] {p} -> format={fmt}")
    return spark.read.format(fmt).load(p)

def _has_data(p):
    return _exists(p) and (read_any(p).limit(1).count() > 0)

def lego_ready(snap):
    pdir = allcat_dir(snap)
    need = {PANEL: f"{pdir}/{PANEL}", FS_SUB: f"{pdir}/{FS_SUB}", BENCH_SUB: f"{pdir}/{BENCH_SUB}"}
    missing = [k for k, p in need.items() if not _exists(p)]
    if missing:
        return False, f"missing {missing}"
    try:
        if read_any(need[BENCH_SUB]).where(F.col(BENCH_COL) > 0).limit(1).count() == 0:
            return False, f"{BENCH_COL} all-zero/null"
    except Exception as e:
        return False, f"{BENCH_COL} unreadable: {str(e)[:50]}"
    return True, "ready"

ready, why = lego_ready(LIVE_SNAPSHOT)
if ready:
    print(f"[{LIVE_SNAPSHOT}] LEGO present -> proceeding to forecast")
else:
    raise RuntimeError(
        f"[{LIVE_SNAPSHOT}] LEGO missing ({why}). Run the FR LEGO pipeline for "
        f"snapshot {LIVE_SNAPSHOT} first.")

# weights from STEP 1 must exist
if not os.path.exists(WEIGHTS_PATH):
    # Try the Volume copy
    vol_copy = f"{PARENT_DIR}/_diq_fit/components/{MARKET}_ensemble_weights.yaml"
    if os.path.exists(vol_copy):
        WEIGHTS_PATH = vol_copy
        print(f"Using Volume copy of weights: {WEIGHTS_PATH}")
    else:
        raise RuntimeError(
            f"Weights not found at {WEIGHTS_PATH}. Run STEP 1 (FR_FitWeights) first, or point "
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
    category_col="LocalLevel4Name",
    config_root=CONFIG_ROOT,
    uk_weights_path=WEIGHTS_PATH, # the fitted per-(category x segment) weights from STEP 1
)
print(f"Forecast rows: {len(stacked):,} | categories: {stacked['category'].nunique()}")
print(f"Written to   : {OUT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE C — write the DIQ stacked forecast into `TF.ml_pred_final`
# MAGIC LEFT-join onto TF so **every row is kept**. `ml_pred_final` = the DIQ stacked
# MAGIC `predicted` where the (key, year_week) is in the DIQ forecast, else the LEGO column
# MAGIC (`LEGO_PRED_COL`, default `prediction_rf`). Set `UPDATE_TF_PRISTINE=true` to enable.

# COMMAND ----------

if not UPDATE_TF:
    print("UPDATE_TF_PRISTINE=false -> skipping the TF update.")
else:
    diq = stacked[["key", "year_week", "predicted"]].copy()
    diq["_dk"] = diq["key"].astype(str)
    diq["_dyw"] = diq["year_week"].astype(str).str.replace("-", "", regex=False)
    diq = (diq.groupby(["_dk", "_dyw"], as_index=False)["predicted"].sum()
              .rename(columns={"predicted": "_diq_pred"}))
    diq_sdf = spark.createDataFrame(diq[["_dk", "_dyw", "_diq_pred"]])

    tf = read_any(TF_PRISTINE)
    n0 = tf.count()
    cols0 = tf.columns
    if "ml_pred_final" not in cols0:
        raise RuntimeError(f"{TF_PRISTINE} has no ml_pred_final column; columns = {cols0}")
    fallback = LEGO_PRED_COL if LEGO_PRED_COL in cols0 else "ml_pred_final"
    print(f"TF rows={n0:,} | DIQ (key×week)={len(diq):,} | unmatched fallback col = '{fallback}'")

    tfj = (tf.withColumn("_k", F.col("key").cast("string"))
             .withColumn("_yw", F.regexp_replace(F.col("year_week").cast("string"), "-", "")))
    j = tfj.join(diq_sdf, (tfj["_k"] == diq_sdf["_dk"]) & (tfj["_yw"] == diq_sdf["_dyw"]), "left")
    j = j.withColumn("ml_pred_final", F.coalesce(F.col("_diq_pred"), F.col(fallback), F.col("ml_pred_final")))
    if "source" in cols0:
        j = j.withColumn("source", F.when(F.col("_diq_pred").isNotNull(),
                                           F.lit("ML_FCST+DIQ_STACK")).otherwise(F.col("source")))
    n_diq = j.where(F.col("_diq_pred").isNotNull()).count()
    result = j.select(cols0)
    n1 = result.count()
    assert n1 == n0, f"ROW COUNT CHANGED {n0} -> {n1} — aborting, NOT writing TF."

    tmp = TF_PRISTINE.rstrip("/") + "_diqtmp"
    result.write.mode("overwrite").parquet(tmp)
    staged = read_any(tmp)
    assert staged.count() == n0, "temp row-count mismatch — aborting before overwrite."
    staged.write.mode("overwrite").parquet(TF_PRISTINE)
    dbutils.fs.rm(tmp, recurse=True)
    print(f"TF updated: {n0:,} rows kept | ml_pred_final = DIQ on {n_diq:,} rows, "
          f"LEGO('{fallback}') on {n0 - n_diq:,}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Peek — the forward forecast (`predicted` = the segment x category x LEGO blend)

# COMMAND ----------

try:
    display(read_any(OUT_PATH).orderBy("category", "key", "year_week").limit(50))
except Exception as e:
    print("spark preview skipped:", e)
    print(stacked.head(20).to_string(index=False))