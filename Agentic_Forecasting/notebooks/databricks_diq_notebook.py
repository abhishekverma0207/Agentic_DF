# Databricks notebook source
# =============================================================================
# DIQ Forward-Forecast Notebook
# =============================================================================
# Copy the contents of this file into a Databricks notebook (or import it as
# a workspace file). The notebook drives a forward-forecast-only run across
# every requested UK category using the unified input table at
# {PARENT_DIR}/DIQ/INPUT_DATA_DELTA.
#
# Prerequisite: this repository must be available on the cluster. Two common
# options:
#   a) Link the repo via Databricks → Repos (clones to
#      /Workspace/Repos/<user>/Agentic_Forecasting), OR
#   b) Upload the folder to /Workspace/Users/<user>/Agentic_Forecasting.
# Either way, set REPO_PATH below to point at the folder.
# =============================================================================

# COMMAND ----------

import os, sys
REPO_PATH = "/Workspace/Repos/<your_email>/Agentic_Forecasting"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets
# MAGIC
# MAGIC - **PARENT_DIR** — root DBFS volume; inputs at `{PARENT_DIR}/{CATS_SUBDIR}` + `{PARENT_DIR}/{FS_SUBDIR}`,
# MAGIC   per-category artifacts under `{PARENT_DIR}/DIQ/{snapshot_week}/{CATEGORY}/`,
# MAGIC   stacked final output at `{PARENT_DIR}/DIQ/TF_DIQ/{snapshot_week}/inference_forecast.parquet`.
# MAGIC - **MARKET** — `uk` or `de`. Picks the YAML template family
# MAGIC   (`config_{market}_{slug}_databricks.yaml`).
# MAGIC - **CATS_SUBDIR** — driver/cats Parquet folder under PARENT_DIR.
# MAGIC   UK: `01_all_keys_dr_adjusted`. DE: `DT`.
# MAGIC - **FS_SUBDIR** — feature-store Parquet folder (defaults to `FS` for both).
# MAGIC - **category_col** — column in the joined input frame carrying the category label.
# MAGIC   UK: `category_name`. DE: `Category`.
# MAGIC - **history_till** — last period of known actuals (YYYYWW), e.g. `202614`.
# MAGIC - **snapshot_week** — first week to forecast (YYYYWW), e.g. `202615`.
# MAGIC - **category_list** — `#`-separated tokens (uppercase with underscores).
# MAGIC - **CONFIG_ROOT** — *(optional)* directory containing the per-category YAML
# MAGIC   templates. Leave **blank** to use the in-repo `<project_root>/config/`
# MAGIC   folder (current behaviour). Set to a UC Volume path like
# MAGIC   `/Volumes/pds_feu_931272_dev/feu_common/config/EU_UK/DIQ_configs/`
# MAGIC   to read templates from that path instead. Filename convention is
# MAGIC   unchanged: `config_{market}_{slug}_databricks.yaml`.

# COMMAND ----------

dbutils.widgets.text(
    "PARENT_DIR",
    "/Volumes/pds_feu_931272_dev/data_science_team/uk_rerun_lego_features",
)
dbutils.widgets.text("MARKET", "uk")
dbutils.widgets.text("CATS_SUBDIR", "01_all_keys_dr_adjusted")
dbutils.widgets.text("FS_SUBDIR", "FS")
dbutils.widgets.text("category_col", "category_name")
dbutils.widgets.text("history_till", "202614")
dbutils.widgets.text("snapshot_week", "202615")
dbutils.widgets.text(
    "category_list",
    "COOKING_AID#CONDIMENT#DEODORANT_FRAGRANCE#FABRIC_CLEANING#"
    "FABRIC_ENHANCER#HAIR_CARE#HOME_HYGIENE#MINI_MEAL#"
    "OTH_FOOD#SKIN_CARE#SKIN_CLEANSING",
)
dbutils.widgets.text("CONFIG_ROOT", "")
# --- v2 deep overlay (TFT for routed high-volume categories) ---
dbutils.widgets.dropdown("DEEP_OVERLAY", "false", ["false", "true"])
dbutils.widgets.text("DEEP_ROUTE", "")  # e.g. FOODS#FABRIC_ENHANCERS#FACE#HAIR_CARE#BODY#DEODORANTS_AND_FRAGRANCES
dbutils.widgets.dropdown("DEEP_MODEL", "TFT", ["TFT", "NHITS", "TiDE"])
dbutils.widgets.text("DEEP_MAX_STEPS", "2000")
# hierarchical XGBoost overlay (cross-account + growth features; beats LEGO on accuracy 8/9 on TH)
dbutils.widgets.dropdown("XGB_OVERLAY", "false", ["false", "true"])
dbutils.widgets.text("XGB_ROUTE", "")  # e.g. FABRIC_CLEANING#HAIR_CARE#FOODS#HOME_AND_HYGIENE#FABRIC_ENHANCERS
dbutils.widgets.dropdown("XGB_OBJ", "huber", ["huber", "ratio", "tweedie"])
dbutils.widgets.dropdown("XGB_PERKEY", "false", ["false", "true"])
dbutils.widgets.text("XGB_ACCOUNT", "E1016")
dbutils.widgets.dropdown("XGB_SEASONAL_TRUST", "true", ["true", "false"])  # leak-free seasonal-level bias fix

# COMMAND ----------

# MAGIC %md
# MAGIC ## MAIN RUN

# COMMAND ----------

def main():
    PARENT_DIR = dbutils.widgets.get("PARENT_DIR")
    MARKET = dbutils.widgets.get("MARKET")
    CATS_SUBDIR = dbutils.widgets.get("CATS_SUBDIR")
    FS_SUBDIR = dbutils.widgets.get("FS_SUBDIR")
    category_col = dbutils.widgets.get("category_col")
    history_till = dbutils.widgets.get("history_till")
    snapshot_week = dbutils.widgets.get("snapshot_week")
    category_list = [c.strip() for c in dbutils.widgets.get("category_list").split("#") if c.strip()]
    # Blank → use in-repo <project_root>/config/. Set a UC Volume path
    # (e.g. /Volumes/pds_feu_931272_dev/feu_common/config/EU_UK/DIQ_configs/)
    # to read the per-category YAMLs from that directory instead.
    CONFIG_ROOT = (dbutils.widgets.get("CONFIG_ROOT") or "").strip() or None
    DEEP_OVERLAY = dbutils.widgets.get("DEEP_OVERLAY").strip().lower() == "true"
    DEEP_ROUTE = [c.strip() for c in dbutils.widgets.get("DEEP_ROUTE").replace("#", ",").split(",") if c.strip()]
    DEEP_MODEL = dbutils.widgets.get("DEEP_MODEL").strip() or "TFT"
    DEEP_MAX_STEPS = int(dbutils.widgets.get("DEEP_MAX_STEPS") or 2000)
    XGB_OVERLAY = dbutils.widgets.get("XGB_OVERLAY").strip().lower() == "true"
    XGB_ROUTE = [c.strip() for c in dbutils.widgets.get("XGB_ROUTE").replace("#", ",").split(",") if c.strip()]
    XGB_OBJ = dbutils.widgets.get("XGB_OBJ").strip() or "huber"
    XGB_PERKEY = dbutils.widgets.get("XGB_PERKEY").strip().lower() == "true"
    XGB_ACCOUNT = dbutils.widgets.get("XGB_ACCOUNT").strip() or "E1016"
    XGB_TRUST = dbutils.widgets.get("XGB_SEASONAL_TRUST").strip().lower() == "true"

    print([PARENT_DIR, MARKET, CATS_SUBDIR, FS_SUBDIR, category_col, history_till, snapshot_week, category_list, CONFIG_ROOT])
    print(f"All DIQ contents are under {PARENT_DIR}/DIQ/")
    if DEEP_OVERLAY:
        print(f"DEEP OVERLAY ON: model={DEEP_MODEL}, steps={DEEP_MAX_STEPS}, route={DEEP_ROUTE}")

    out_path = f"{PARENT_DIR}/DIQ/TF_DIQ/{category_list[0]}_inference_forecast.parquet"

    from utils.diq_runner import run_diq_forecast
    stacked = run_diq_forecast(
        parent_dir=PARENT_DIR,
        history_till=history_till,
        snapshot_week=snapshot_week,
        category_list=category_list,
        out_path=out_path,
        market=MARKET,
        cats_subdir=CATS_SUBDIR,
        fs_subdir=FS_SUBDIR,
        category_col=category_col,
        config_root=CONFIG_ROOT,
        enable_deep_overlay=DEEP_OVERLAY,
        deep_route_categories=DEEP_ROUTE,
        deep_model=DEEP_MODEL,
        deep_max_steps=DEEP_MAX_STEPS,
        enable_xgb_overlay=XGB_OVERLAY,
        xgb_route_categories=XGB_ROUTE,
        xgb_obj=XGB_OBJ,
        xgb_perkey=XGB_PERKEY,
        xgb_target_account=XGB_ACCOUNT,
        xgb_seasonal_trust=XGB_TRUST,
        # continue_on_category_failure=True,
    )

    print(f"Forecast saved at {out_path}")
    print(f"Rows: {len(stacked)}")
    print(f"Categories: {stacked['category'].nunique()}")
    moq_rewrites = int((stacked['prediction_post_moq'] != stacked['predicted']).sum())
    print(f"MOQ-rewritten rows: {moq_rewrites}")
    print("Run completed")


main()
