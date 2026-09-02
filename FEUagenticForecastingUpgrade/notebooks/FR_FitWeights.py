# Databricks notebook source
# =============================================================================
# FR_FitWeights — STEP 1 of 2: calculate the ensemble weights for France
# =============================================================================
# Mirrors DE_FitWeights exactly, adapted for France's Volume paths and config.
#
# France has the SAME DT/FS structure as DE, so uses parent_dir_template
# (production-fidelity DT⋈FS join) — no data preparation step needed.
# Benchmark subfolder is TF (not TF_pristine_allkeys like DE).
# LEGO prediction column is prediction_rf (not prediction_xgb like DE).
# RUN_LEGO_IF_MISSING defaults to false (no LEGO auto-run notebook for FR).
#
# For each TRAINING snapshot S:
#   * LEGO must already be present (DT/, FS/, TF with prediction_rf)
#   * run the NEW DIQ (segment components) and save them
# Then pool every snapshot + realized actuals and fit the per-(category x lego_segment)
# weights -> config/fr_ensemble_weights.yaml
#
# STEP 2 = FR_Forecast: takes the weights this writes and produces the forward forecast.
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

# ---- FR config ----
dbutils.widgets.text("PARENT_DIR",       "/Volumes/pds_feu_931272_dev/eu_france/transform/FR/DIQ_Test/lego_output")
dbutils.widgets.text("TRAIN_SNAPSHOTS",  "202603,202604,202607,202612")
dbutils.widgets.text("LIVE_SNAPSHOT",    "202616")                         # EXCLUDED from the fit
dbutils.widgets.text("ACTUALS_SNAPSHOT", "202629")                         # DT supplies realized forward actuals
dbutils.widgets.text("ENGINE_WORKERS",   "6")            # ProcessPool workers on the DRIVER
dbutils.widgets.text("ENGINE_THREADS",   "4")            # LightGBM threads / worker
dbutils.widgets.text("HORIZONS",         "13")           # forward weeks per snapshot
dbutils.widgets.text("MIN_FWD_WEEKS",    "5")            # min realized forward weeks to include a snapshot
dbutils.widgets.text("OBJECTIVE",        "wape_asym")
dbutils.widgets.text("BIAS_TOLERANCE",   "0.02")
dbutils.widgets.dropdown("GUARDRAIL",    "false", ["true", "false"])
dbutils.widgets.text("SEG_SHRINK_K",     "0")            # 0 = raw per-segment coefficients
dbutils.widgets.text("SEG_MIN_VOLUME",   "0")            # 0 = fit even low-volume segments

W = lambda k: dbutils.widgets.get(k).strip()
PARENT_DIR        = W("PARENT_DIR").rstrip("/")
LIVE_SNAPSHOT     = W("LIVE_SNAPSHOT")
ACTUALS_SNAPSHOT  = W("ACTUALS_SNAPSHOT")
HORIZONS          = int(W("HORIZONS"))
MIN_FWD_WEEKS     = int(W("MIN_FWD_WEEKS"))
OBJECTIVE         = W("OBJECTIVE")
BIAS_TOLERANCE    = float(W("BIAS_TOLERANCE"))
GUARDRAIL         = W("GUARDRAIL").lower() == "true"
os.environ["UK_ENGINE_WORKERS"] = W("ENGINE_WORKERS")
os.environ["UK_ENGINE_THREADS"] = W("ENGINE_THREADS")
os.environ["SEG_SHRINK_K"]      = W("SEG_SHRINK_K")
os.environ["SEG_MIN_VOLUME"]    = W("SEG_MIN_VOLUME")

# FR conventions (mirror config_fr_base_local.py market_io)
MARKET, PANEL, FS_SUB = "fr", "DT", "FS"
BENCH_SUB, BENCH_COL, ACTUALS_COL = "TF", "prediction_rf", "NonPromoVolValue"
CONFIG_ROOT     = os.path.join(REPO_PATH, "config")
COMPONENTS_OUT  = f"{PARENT_DIR}/_diq_fit/components"
allcat_dir      = lambda s: f"{PARENT_DIR}/{s}"
PARENT_DIR_TMPL = f"{PARENT_DIR}/{{snapshot}}"
ACTUALS_SOURCE  = f"{allcat_dir(ACTUALS_SNAPSHOT)}/{PANEL}"

os.environ["DIQ_RUNNER_SKIP_INSTALL"] = "1"

import pyspark.sql.functions as F
import pandas as pd
from utils import weight_fit as wf
from utils.uk_forecast import add_weeks
print("MARKET:", MARKET, "| PARENT_DIR:", PARENT_DIR)
print("CONFIG_ROOT:", CONFIG_ROOT, "| COMPONENTS_OUT:", COMPONENTS_OUT)

# COMMAND ----------

print("REPO_PATH resolved to:", REPO_PATH)
import os; print("utils exists?", os.path.isdir(os.path.join(REPO_PATH, "utils")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE 0 — readiness: which training snapshots have LEGO + enough realized actuals

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd

# ---------- format-aware readers ----------
def _ls(p):
    try:
        return list(dbutils.fs.ls(p))
    except Exception:
        return []

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
    """Read a path without caring whether it was written as Delta or parquet.
    FR's DT panel is a Delta table; FS/TF may still be plain parquet."""
    fmt = _fmt(p)
    print(f"  [read_any] {p} -> format={fmt}")
    return spark.read.format(fmt).load(p)

def _has_data(p):
    """True if p holds a readable, non-empty table (Delta or parquet)."""
    if not _exists(p):
        return False
    if _fmt(p) == "delta":
        try:
            return read_any(p).limit(1).count() > 0
        except Exception:
            return False
    return any(f.name.endswith(".parquet") or f.name.startswith("part-") or f.isDir()
               for f in _ls(p))


# ---------- readiness check ----------
def lego_ready(snap):
    """All LEGO requirements present for this snapshot's All_Cat (DT/, FS/, TF with prediction_rf)?"""
    pdir = allcat_dir(snap)
    need = {PANEL: f"{pdir}/{PANEL}", FS_SUB: f"{pdir}/{FS_SUB}", BENCH_SUB: f"{pdir}/{BENCH_SUB}"}
    missing = [k for k, p in need.items() if not _has_data(p)]
    if missing:
        return False, f"missing {missing}"
    try:
        if read_any(need[BENCH_SUB]).where(F.col(BENCH_COL) > 0).limit(1).count() == 0:
            return False, f"{BENCH_COL} all-zero/null"
    except Exception as e:
        return False, f"{BENCH_COL} unreadable: {str(e)[:50]}"
    return True, "ready"


# ---------- how far actuals are realized ----------
_am = (read_any(ACTUALS_SOURCE)
       .where(F.col(ACTUALS_COL) > 0)
       .agg(F.max(F.regexp_replace(F.col("year_week").cast("string"), "-", "")))
       .first()[0])
if _am is None:
    raise ValueError(f"No positive {ACTUALS_COL} rows found in {ACTUALS_SOURCE}")
actuals_max = int(str(_am))

print("Actuals realized through:", actuals_max, f"(from {ACTUALS_SNAPSHOT})")


# ---------- snapshot selection ----------
explicit = [s.strip() for s in W("TRAIN_SNAPSHOTS").replace(",", "#").split("#") if s.strip()]
rows, fit_snaps = [], []
for snap in explicit:
    covered    = sum(1 for h in range(HORIZONS) if int(add_weeks(snap, h)) <= actuals_max)
    ready, why = lego_ready(snap)
    include    = ready and covered >= MIN_FWD_WEEKS
    rows.append({"snapshot": snap, "lego_ready": ready, "note": why,
                 "fwd_weeks_covered": covered, "include_in_fit": include})
    if include:
        fit_snaps.append(snap)

try:
    display(pd.DataFrame(rows))
except Exception:
    print(pd.DataFrame(rows).to_string(index=False))

print("\nFIT snapshots:", fit_snaps)
assert fit_snaps, "No snapshot is fit-ready (need LEGO present + >= MIN_FWD_WEEKS realized actuals)."

# COMMAND ----------

# def _ls(p):
#     try: return list(dbutils.fs.ls(p))
#     except Exception: return []

# def _has_parquet(p):
#     return any(f.name.endswith(".parquet") or f.name.startswith("part-") or f.isDir() for f in _ls(p))

# def lego_ready(snap):
#     """All LEGO requirements present for this snapshot's All_Cat (DT/, FS/, TF with prediction_rf)?"""
#     pdir = allcat_dir(snap)
#     need = {PANEL: f"{pdir}/{PANEL}", FS_SUB: f"{pdir}/{FS_SUB}", BENCH_SUB: f"{pdir}/{BENCH_SUB}"}
#     missing = [k for k, p in need.items() if not _has_parquet(p)]
#     if missing:
#         return False, f"missing {missing}"
#     try:
#         if spark.read.parquet(need[BENCH_SUB]).where(F.col(BENCH_COL) > 0).limit(1).count() == 0:
#             return False, f"{BENCH_COL} all-zero/null"
#     except Exception as e:
#         return False, f"{BENCH_COL} unreadable: {str(e)[:50]}"
#     return True, "ready"

# actuals_max = int(str(
#     spark.read.parquet(ACTUALS_SOURCE).where(F.col(ACTUALS_COL) > 0)
#     .agg(F.max(F.regexp_replace(F.col("year_week").cast("string"), "-", ""))).first()[0]))
# print("Actuals realized through:", actuals_max, f"(from {ACTUALS_SNAPSHOT})")

# explicit   = [s.strip() for s in W("TRAIN_SNAPSHOTS").replace(",", "#").split("#") if s.strip()]
# rows, fit_snaps = [], []
# for snap in explicit:
#     covered    = sum(1 for h in range(HORIZONS) if int(add_weeks(snap, h)) <= actuals_max)
#     ready, why = lego_ready(snap)
#     include    = ready and covered >= MIN_FWD_WEEKS
#     rows.append({"snapshot": snap, "lego_ready": ready, "note": why,
#                  "fwd_weeks_covered": covered, "include_in_fit": include})
#     if include:
#         fit_snaps.append(snap)

# try:    display(pd.DataFrame(rows))
# except Exception: print(pd.DataFrame(rows).to_string(index=False))
# print("\nFIT snapshots:", fit_snaps)
# assert fit_snaps, "No snapshot is fit-ready (need LEGO present + >= MIN_FWD_WEEKS realized actuals)."

# COMMAND ----------

# MAGIC %md
# MAGIC ## STAGE A — save the DIQ components for each training snapshot
# MAGIC
# MAGIC LEGO must already be present (DT/, FS/, TF with prediction_rf).
# MAGIC RUN_LEGO_IF_MISSING is not supported for FR.

# COMMAND ----------

saved = []
for snap in fit_snaps:
    ready, why = lego_ready(snap)
    if not ready:
        raise RuntimeError(
            f"[{snap}] LEGO missing ({why}). Run the FR LEGO pipeline for this snapshot first.")
    info = wf.save_components(
        MARKET, snap,
        out_dir=COMPONENTS_OUT,
        config_root=CONFIG_ROOT,
        parent_dir_template=PARENT_DIR_TMPL,                  # DT ⋈ FS panel, exactly like LIVE
        lego_path_template=f"{PARENT_DIR_TMPL}/{BENCH_SUB}",  # prediction_rf benchmark
        in_scope_categories=None,                             # route all categories
        horizons=HORIZONS,
    )
    saved.append(info["path"])
    print(f"  [A] {snap}: {info['rows']:>8,} rows, {len(info['categories'])} cats -> {info['path']}")
print(f"\nStage A complete: {len(saved)} snapshots' components saved.")

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

# Volume copy so the weights survive an ephemeral job cluster.
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
                   "w_seg": wg, "w_cat": wl, "w_xgb": wx,
                   "acc_xgb": v.get("acc_lego"), "acc_stack": v.get("acc_stack"),
                   "acc_delta": round(v.get("acc_stack", 0) - v.get("acc_lego", 0), 4),
                   "bias_xgb": v.get("bias_lego"), "bias_stack": v.get("bias_stack")})
    except Exception:
        sb.append({"group": str(key), "use": str(v)[:40]})
score = pd.DataFrame(sb)
n_stack = int((score.get("use") == "STACK").sum()) if "use" in score else 0
print(f"STACK deployed for {n_stack}/{len(score)} (category x segment) groups (rest fall back to pure prediction_rf).")
try:    display(score)
except Exception: print(score.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done — weights fitted. NEXT: STEP 2 = `FR_Forecast`
# MAGIC
# MAGIC `config/fr_ensemble_weights.yaml` is written (and copied to the Volume,
# MAGIC printed above as `VOLUME_WEIGHTS`). Run **`FR_Forecast`** next to apply these
# MAGIC weights and produce the forward forecast — point its `WEIGHTS_PATH` at either
# MAGIC the committed `config/fr_ensemble_weights.yaml` or the Volume copy above.