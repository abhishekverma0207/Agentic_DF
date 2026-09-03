# Databricks notebook source
# =============================================================================
# LEGO_RUN_DE_PARAM — parametrized DE LEGO entry (for DIQ_FitWeights auto-trigger)
# =============================================================================
# A widget-driven clone of DE_AllCAT_PARALLEL_RUN. DIQ_FitWeights calls it via
# dbutils.notebook.run(LEGO_RUN_NOTEBOOK, {HISTORY_CUT_OFF, SNAPSHOT_WEEK,
# DEMANDIQ_RUN_FLAG, PARENT_DIR, ...}) to regenerate LEGO for a snapshot whose
# artifacts are missing, then runs its own new DIQ + fit.
#
# IMPORTANT — placement: this notebook uses RELATIVE %run paths (../../Config,
# ../LEGO_MODULES/..., ../../Main_Class, ../CONFIG_DEFAULT_DE, ./LEGO_RUN_SCRIPT_DE),
# so it MUST be imported into the SAME workspace folder as DE_AllCAT_PARALLEL_RUN:
#   /Workspace/Users/anu.thomas@unilever.com/DACH_Industrialized/ParallelRun_Scripts/DE/
# Point DIQ_FitWeights' LEGO_RUN_NOTEBOOK widget at this notebook's workspace path.
#
# What it runs: DT → FE → FS → TF (LEGO XGB → prediction_xgb) → Combine → TF_pristine_allkeys.
# With DEMANDIQ_RUN_FLAG=false it SKIPS the old per-category DIQ job fan-out (DIQ_FitWeights
# runs its own new DIQ next) — so it produces just the panel + benchmark the fit needs.
#
# CAVEAT — point-in-time data: regenerating a HISTORICAL snapshot requires the raw
# modelling data for that week to be available. Set MODELLING_DATA_PATH to the dump that
# corresponds to HISTORY_CUT_OFF. For the existing parallel_run snapshots LEGO already
# exists, so DIQ_FitWeights skips the trigger entirely — this wrapper is the safety net.
# =============================================================================

# COMMAND ----------

!pip install azure-keyvault-secrets azure-identity azure-keyvault-keys opencensus-ext-azure isoweek

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widgets (per-snapshot inputs)
dbutils.widgets.text("COUNTRY_CODE", "DE")
dbutils.widgets.text("SALES_ORG", "1300")
dbutils.widgets.text("COUNTRY_NAME", "Germany")
dbutils.widgets.text("CATEGORY_SCHEMA", "BPC")        # the 'category' label arg to ParallelRun
dbutils.widgets.text("CATEGORIES", "")                # '#'-separated; blank = all categories
dbutils.widgets.text("HISTORY_CUT_OFF", "2026-04")    # YYYY-WW or YYYYWW (normalized below)
dbutils.widgets.text("SNAPSHOT_WEEK", "")             # blank → increment_week(history_cut_off, 1)
dbutils.widgets.dropdown("DEMANDIQ_RUN_FLAG", "true", ["true", "false"])
dbutils.widgets.text("PARENT_DIR", "")                # blank → default DIQ_Test_v1 path; else exact All_Cat dir
dbutils.widgets.text("MODELLING_DATA_PATH",
    "/Volumes/pds_feu_931272_dev/eu_dach/landing/DACH/DACH_LEGO/processed_data/modelling_data/202615/20260406/")

# COMMAND ----------

import collections

debug = False
is_develop_mountpoint = False
country_Code = dbutils.widgets.get("COUNTRY_CODE").strip()
sales_Org    = dbutils.widgets.get("SALES_ORG").strip()
country_name = dbutils.widgets.get("COUNTRY_NAME").strip()
category     = dbutils.widgets.get("CATEGORY_SCHEMA").strip()
categories   = [c.strip() for c in dbutils.widgets.get("CATEGORIES").replace(",", "#").split("#") if c.strip()]
demandiq_run_flag = dbutils.widgets.get("DEMANDIQ_RUN_FLAG").strip().lower() == "true"
MODELLING_DATA_PATH = dbutils.widgets.get("MODELLING_DATA_PATH").strip()
PARENT_DIR_W = dbutils.widgets.get("PARENT_DIR").strip()

# Normalize history_cut_off to the DE hyphenated year-week format ('YYYY-WW').
_h = dbutils.widgets.get("HISTORY_CUT_OFF").strip().replace("-", "")
history_cut_off = f"{_h[:4]}-{_h[4:]}"
print(["country_Code", country_Code, "sales_Org", sales_Org, "categories", categories,
       "history_cut_off", history_cut_off, "demandiq_run_flag", demandiq_run_flag])

# COMMAND ----------

# DBTITLE 1,Config
# MAGIC %run ../../Config

# COMMAND ----------

# DBTITLE 1,Common Import
# MAGIC %run ../LEGO_MODULES/version=1.5/Common_Import

# COMMAND ----------

# DBTITLE 1,Main Class
# MAGIC %run ../../Main_Class

# COMMAND ----------

# DBTITLE 1,Base Path & User Inputs (parametrized)
# snapshot_week: explicit widget (normalized) or computed from history_cut_off, like the original.
_s = dbutils.widgets.get("SNAPSHOT_WEEK").strip().replace("-", "")
snapshot_week = f"{_s[:4]}-{_s[4:]}" if _s else increment_week(history_cut_off, 1)

# Output root. When DIQ_FitWeights triggers us it passes PARENT_DIR = the exact {snap}/All_Cat
# dir it will read from, so LEGO lands precisely there. Else fall back to the default location.
if PARENT_DIR_W:
    path_base = PARENT_DIR_W if PARENT_DIR_W.endswith("/") else PARENT_DIR_W + "/"
elif len(categories) == 0:
    path_base = f'/Volumes/pds_feu_931272_dev/eu_dach/landing/Experiments/DIQ_Test_v1/Germany/{snapshot_week}/All_Cat/'
else:
    path_base = f'/Volumes/pds_feu_931272_dev/eu_dach/landing/Experiments/DIQ_Test_v1/Germany/{snapshot_week}/{category}/'

modelling_data_writing_paths['Actuals']['modelling_data'] = MODELLING_DATA_PATH
print([history_cut_off, snapshot_week, path_base])

# COMMAND ----------

# DBTITLE 1,Get Required Input Data
try:
  data = get_modelling_data_with_apg_product()
  logger.info("Success! Germany get_modelling_data PASSED")
except:
  logger.error("Warning! Germany get_modelling_data FAILED")
  raise Exception()

# COMMAND ----------

# DBTITLE 1,Instantiating Parallel Run Main Class
try:
  ParallelRun_Obj = ParallelRun(filters, country_Code, sales_Org, category, categories, data)
  logger.info("Success! Germany ParallelRun Class Instantiating PASSED")
except:
  logger.error("Warning! Germany ParallelRun Class Instantiating FAILED")
  raise Exception()

# COMMAND ----------

# DBTITLE 1,Process Modelling Data
try:
  modelling_data = ParallelRun_Obj.get_processed_modelling_data_de()
  logger.info("Success! Germany get_processed_modelling_data PASSED")
except:
  logger.error("Warning! Germany get_processed_modelling_data FAILED")
  raise Exception()

# COMMAND ----------

modelling_data = add_week_of_promo(modelling_data, "None", "N")
modelling_data = modelling_data.withColumn("PromoStart", F.when(F.col("Promo_Week") == 1, 1).otherwise(0))
modelling_data = modelling_data.withColumn("PromoSecond", F.when(F.col("Promo_Week") == 2, 1).otherwise(0))

promo_cols = [col for col in modelling_data.columns if ("promo" in col.lower())]
promo_cols = [col for col in promo_cols if col not in ["PromoID", "PromoFlag", "PromoStart", "Promo_Week", "PromoSecond"]]

for col in ["Duration", "Promo_Week", "PromoStart", "PromoSecond"]:
  modelling_data = modelling_data.withColumn(col,
                                             F.when(F.col("Category").isin(["HEALTHY SNACKING", "DRESSING"]),
                                              0).otherwise(F.col(col)))

# COMMAND ----------

# MAGIC %run ../CONFIG_DEFAULT_DE

# COMMAND ----------

# DBTITLE 1,CONFIG Update
lego_config['sales']['history_till'] = history_cut_off
print([lego_config['sales']['history_till'], snapshot_week])

# COMMAND ----------

modelling_data = modelling_data\
                .withColumn('total_holidays', reduce(operator.add, [F.col(c) for c in holiday_features]))\
                .withColumn("HolidayFlag",
                            F.when(F.col("total_holidays") > 0,
                                   1).otherwise(0))\
              .withColumn("CPL", F.split(F.col("key"), "_").getItem(1))

# COMMAND ----------

# DBTITLE 1,Config Validation
config_class.config_data_validation(SparkDF=modelling_data, config_dict=lego_config, parallel_run=True, backtesting_run=False, return_in_fail=False)

# COMMAND ----------

# DBTITLE 1,Data Filter (sales org)
modelling_data = modelling_data.filter((F.col('SalesOrg') == sales_Org))

# COMMAND ----------

# DBTITLE 1,Enabling for all category (XGB benchmark)
lego_config['forecasting']['total_forecast']['XGB'] = True
lego_config['forecasting']['total_forecast']['RF']  = False

modelling_data_c = modelling_data

# COMMAND ----------

# DBTITLE 1,PATH
parentDir = f'{path_base}'
print(parentDir)
DT_PATH = parentDir + 'DT/'
FE_PATH = parentDir + 'FE/'
FS_PATH = parentDir + 'FS/'
TF_PATH = parentDir + 'TF/'
BASELINE_PATH = parentDir + 'Baseline/'
legotag_path = parentDir + 'Lego_Tags/'

# COMMAND ----------

# DBTITLE 1,Category wise count
modelling_data_c.groupby('Category').agg(F.countDistinct('key').alias('key_cnt')).display()

# COMMAND ----------

# DBTITLE 1,Run LEGO (DT→FE→FS→TF→Combine; old DIQ skipped when DEMANDIQ_RUN_FLAG=false)
# MAGIC %run ./LEGO_RUN_SCRIPT_DE

# COMMAND ----------

# DBTITLE 1,Verify benchmark written
try:
    n = spark.read.parquet(parentDir + 'TF_pristine_allkeys').where(F.col('prediction_xgb') > 0).limit(1).count()
    print(f"TF_pristine_allkeys prediction_xgb populated: {'YES' if n else 'NO'}  at {parentDir}TF_pristine_allkeys")
except Exception as e:
    print("Could not verify TF_pristine_allkeys:", e)
