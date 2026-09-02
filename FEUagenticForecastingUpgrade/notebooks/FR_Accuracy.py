# Databricks notebook source
# =============================================================================
# FR_Accuracy — DIQ stacked vs LEGO benchmark vs Actuals
# =============================================================================
# Loops over multiple live snapshots, joins DIQ + LEGO + Actuals at GTIN level,
# calculates abs_error, unions everything into one table for Excel pivot.
# =============================================================================

import pyspark.sql.functions as F
import pandas as pd

def _exists(p):
    try:
        dbutils.fs.ls(p)
        return True
    except Exception:
        return False

def _fmt(p):
    return "delta" if _exists(f"{p}/_delta_log") else "parquet"

def read_any(p):
    fmt = _fmt(p)
    print(f"  [read_any] {p} -> format={fmt}")
    return spark.read.format(fmt).load(p)

# COMMAND ----------


# df_chk = spark.read.format('delta').load("/Volumes/pds_feu_931272_dev/eu_france/transform/FR/DIQ_Test/lego_output/202629/FS")
# df = df_chk.toPandas()
# df.head(2)
sorted([col for col in df.columns if col.startswith("f_")])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

PARENT_DIR        = "/Volumes/pds_feu_931272_dev/eu_france/transform/FR/DIQ_Test/lego_output"
LEGO_PRED_COL     = "prediction_rf"
ACTUALS_COL       = "NonPromoVolValue"
KEY_COL           = "key"
WEEK_COL          = "year_week"
LIVE_SNAPSHOT_LIST = ["202616", "202620"]
ACTUALS_SNAPSHOT  = "202629"
ACTUALS_PATH      = f"{PARENT_DIR}/{ACTUALS_SNAPSHOT}/DT"

print("=" * 60)
print("FR ACCURACY EVALUATION")
print("=" * 60)
print(f"  Snapshots to evaluate : {LIVE_SNAPSHOT_LIST}")
print(f"  Actuals snapshot      : {ACTUALS_SNAPSHOT}")
print(f"  Actuals path          : {ACTUALS_PATH}")
print(f"  LEGO pred col         : {LEGO_PRED_COL}")
print(f"  Actuals col           : {ACTUALS_COL}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Actuals (once — shared across all snapshots)

# COMMAND ----------

# df_chk = spark.read.format('delta').load('/Volumes/pds_feu_931272_dev/eu_france/transform/FR/FR_LEGO/France/parallel_run1/202629/DT')
# # sorted(df_chk.columns)
# # display(df_chk.select(F.col('CategoryDescription')).distinct())
# # display(df_chk.groupby(['CategoryDescription', 'BrandDescription', 'LocalLevel4Name']).agg(F.countDistinct('key')))
# display(df_chk.groupby(['LocalLevel4Name']).agg(F.countDistinct('key')))

# COMMAND ----------

print(f"\n--- Loading Actuals from {ACTUALS_SNAPSHOT} ---")
act_sdf = read_any(ACTUALS_PATH)
print(f"  Actuals DT rows: {act_sdf.count():,}")

# Normalise year_week
act_sdf = act_sdf.withColumn(WEEK_COL, F.regexp_replace(F.col(WEEK_COL).cast("string"), "-", ""))

# Keep key-level actuals for joining (aggregate to GTIN happens AFTER join)
act_key = (act_sdf
    .groupBy(KEY_COL, WEEK_COL)
    .agg(F.sum(ACTUALS_COL).alias("actual"))
)
print(f"  Actuals at key level: {act_key.count():,} rows")

# COMMAND ----------

# display(
#     act_sdf.filter(
#         (F.col('year_week') <= 202629) &
#         (F.col('LocalLevel4Name').isin(['CONDIMENT']))
#     )
# )

# display(
#     act_sdf.filter(
#         (F.col('year_week') <= 202628) & (F.col('year_week') >= 202615)
#     )
#     .groupBy('LocalLevel4Name', 'year_week')
#     .agg(F.sum('NonPromoVolValue'))
#     .orderBy('LocalLevel4Name', 'year_week')
# )

# COMMAND ----------

# ── Filter out keys that first appear after the earliest eval week ──
# Eval week = earliest snapshot + horizon (5 weeks)
first_snap = min(int(s) for s in LIVE_SNAPSHOT_LIST)
first_snap_year = first_snap // 100
first_snap_wk = first_snap % 100
eval_wk = first_snap_wk + 4  # horizon - 1 = 4 weeks ahead
eval_year = first_snap_year
if eval_wk > 52:
    eval_wk -= 52
    eval_year += 1
FIRST_EVAL_WEEK = str(eval_year * 100 + eval_wk)

first_week_per_key = (
    act_key
    .groupBy(KEY_COL)
    .agg(F.min(WEEK_COL).alias("first_week"))
)
late_keys = first_week_per_key.filter(F.col("first_week") > FIRST_EVAL_WEEK)
n_late = late_keys.count()
n_before = act_key.select(KEY_COL).distinct().count()

act_key = act_key.join(late_keys.select(KEY_COL), on=KEY_COL, how="left_anti")
print(f"  First eval week: {FIRST_EVAL_WEEK}")
print(f"  Removed {n_late:,} keys starting after {FIRST_EVAL_WEEK} ({n_before:,} → {act_key.select(KEY_COL).distinct().count():,} keys)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Loop over snapshots — load DIQ + LEGO, join, union

# COMMAND ----------

all_results = []

for i, LIVE_SNAPSHOT in enumerate(LIVE_SNAPSHOT_LIST, 1):
    print(f"\n{'=' * 60}")
    print(f"  SNAPSHOT {i}/{len(LIVE_SNAPSHOT_LIST)}: {LIVE_SNAPSHOT}")
    print(f"{'=' * 60}")

    DIQ_PATH  = f"{PARENT_DIR}/{LIVE_SNAPSHOT}/DIQ/TF_DIQ/fr_{LIVE_SNAPSHOT}_inference_forecast.parquet"
    LEGO_PATH = f"{PARENT_DIR}/{LIVE_SNAPSHOT}/TF"

    # --- 2a. Load DIQ forecast at key level ---
    print(f"\n  [DIQ] {DIQ_PATH}")
    diq_sdf = read_any(DIQ_PATH)
    diq_count = diq_sdf.count()
    print(f"  [DIQ] raw rows: {diq_count:,}")

    diq_sdf = diq_sdf.withColumn(WEEK_COL, F.regexp_replace(F.col(WEEK_COL).cast("string"), "-", ""))

    # Filter to horizon=5 at key level (before any aggregation)
    diq_sdf = diq_sdf.filter(F.col("horizon") <= 13)
    diq_h5_count = diq_sdf.count()
    print(f"  [DIQ] after horizon=5 filter: {diq_h5_count:,} rows")

    # Keep key-level DIQ: key, year_week, predicted, category
    diq_key = diq_sdf.select(KEY_COL, WEEK_COL, "predicted", "category")

    # --- 2b. Load LEGO benchmark at key level ---
    print(f"\n  [LEGO] {LEGO_PATH}")
    lego_sdf = read_any(LEGO_PATH)
    lego_count = lego_sdf.count()
    print(f"  [LEGO] raw rows: {lego_count:,}")

    lego_sdf = lego_sdf.withColumn(WEEK_COL, F.regexp_replace(F.col(WEEK_COL).cast("string"), "-", ""))

    # Filter to Forecast rows only, keep key level
    lego_sdf = lego_sdf.filter(F.col("Period_Type") == "Forecast")
    lego_fc_count = lego_sdf.count()
    print(f"  [LEGO] after Period_Type='Forecast' filter: {lego_fc_count:,} rows")

    lego_key = lego_sdf.select(KEY_COL, WEEK_COL, F.col(LEGO_PRED_COL).alias("lego_predicted"))

    # --- 2c. Join all three at KEY level first ---
    print(f"\n  [JOIN] key-level join: DIQ left-join LEGO left-join Actuals on ({KEY_COL}, {WEEK_COL})")
    joined = (diq_key
        .join(lego_key, on=[KEY_COL, WEEK_COL], how="left")
        .join(act_key, on=[KEY_COL, WEEK_COL], how="left")
    )
    joined_count = joined.count()
    print(f"  [JOIN] key-level joined rows: {joined_count:,}")

    # Drop rows where actuals or lego are missing
    joined = joined.filter(F.col("actual").isNotNull())
    joined = joined.filter(F.col("lego_predicted").isNotNull())
    filtered_count = joined.count()
    dropped = joined_count - filtered_count
    print(f"  [JOIN] after dropping null actual/lego: {filtered_count:,} rows (dropped {dropped:,})")

    # --- 2d. Derive cs_gtin and aggregate to GTIN x year_week ---
    joined = joined.withColumn("cs_gtin", F.regexp_replace(F.col(KEY_COL).cast("string"), "_[^_]+$", ""))

    gtin_agg = (joined
        .groupBy("cs_gtin", WEEK_COL, "category")
        .agg(
            F.sum("predicted").alias("diq_predicted"),
            F.sum("lego_predicted").alias("lego_predicted"),
            F.sum("actual").alias("actual"),
        )
    )
    gtin_agg = gtin_agg.withColumn("snapshot_week", F.lit(LIVE_SNAPSHOT))
    gtin_count = gtin_agg.count()
    print(f"  [AGG] aggregated to GTIN x year_week: {gtin_count:,} rows")

    # --- 2e. Calculate errors at GTIN level ---
    snap_result = (gtin_agg
        .withColumn("abs_error_diq",  F.abs(F.col("diq_predicted") - F.col("actual")))
        .withColumn("abs_error_lego", F.abs(F.col("lego_predicted") - F.col("actual")))
        .select(
            "category", "cs_gtin", "year_week", "snapshot_week",
            "actual",
            "diq_predicted", "lego_predicted",
            "abs_error_diq", "abs_error_lego",
        )
    )
    all_results.append(snap_result)
    print(f"  [DONE] snapshot {LIVE_SNAPSHOT}: {gtin_count:,} GTIN rows added to result")

# --- Union all snapshots ---
print(f"\n{'=' * 60}")
print(f"  UNION: combining {len(all_results)} snapshot(s)")
print(f"{'=' * 60}")

from functools import reduce
result = reduce(lambda a, b: a.unionByName(b), all_results)
result = result.orderBy("snapshot_week", "category", "cs_gtin", "year_week")
total_rows = result.count()
print(f"  Total result rows: {total_rows:,}")
print(f"  Snapshots: {result.select('snapshot_week').distinct().collect()}")

# COMMAND ----------

# display(
#     result.select('snapshot_week', 'year_week').dropDuplicates().orderBy('snapshot_week', 'year_week')
# )

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Overall summary stats

# COMMAND ----------

print("\n" + "=" * 60)
print("OVERALL ACCURACY (1 - WAPE)")
print("=" * 60)

summary = result.agg(
    F.count("*").alias("n_rows"),
    F.countDistinct("cs_gtin").alias("n_gtins"),
    F.countDistinct("year_week").alias("n_weeks"),
    F.countDistinct("category").alias("n_categories"),
    F.countDistinct("snapshot_week").alias("n_snapshots"),
    F.sum("actual").alias("total_actual"),
    F.sum("diq_predicted").alias("total_diq"),
    F.sum("lego_predicted").alias("total_lego"),
    F.sum("abs_error_diq").alias("total_abs_error_diq"),
    F.sum("abs_error_lego").alias("total_abs_error_lego"),
)
summary_row = summary.first()

total_actual = summary_row["total_actual"] or 1e-9
wape_diq  = summary_row["total_abs_error_diq"]  / total_actual
wape_lego = summary_row["total_abs_error_lego"] / total_actual
acc_diq   = 1 - wape_diq
acc_lego  = 1 - wape_lego
bias_diq  = (summary_row["total_diq"]  - total_actual) / total_actual
bias_lego = (summary_row["total_lego"] - total_actual) / total_actual

print(f"  DIQ stacked  : acc={acc_diq:.4f}  WAPE={wape_diq:.4f}  bias={bias_diq:+.4f}")
print(f"  LEGO baseline: acc={acc_lego:.4f}  WAPE={wape_lego:.4f}  bias={bias_lego:+.4f}")
print(f"  Delta (DIQ-LEGO): {acc_diq - acc_lego:+.4f}")
print(f"  ---")
print(f"  Snapshots: {summary_row['n_snapshots']}  |  GTINs: {summary_row['n_gtins']}  |  Weeks: {summary_row['n_weeks']}  |  Categories: {summary_row['n_categories']}")

try:
    display(summary)
except Exception:
    print(summary.toPandas().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Per-snapshot accuracy

# COMMAND ----------

snap_stats = (result
    .groupBy("snapshot_week")
    .agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("cs_gtin").alias("n_gtins"),
        F.sum("actual").alias("total_actual"),
        F.sum("diq_predicted").alias("total_diq"),
        F.sum("lego_predicted").alias("total_lego"),
        F.sum("abs_error_diq").alias("abs_err_diq"),
        F.sum("abs_error_lego").alias("abs_err_lego"),
    )
    .withColumn("wape_diq",  F.col("abs_err_diq")  / F.greatest(F.col("total_actual"), F.lit(1e-9)))
    .withColumn("wape_lego", F.col("abs_err_lego") / F.greatest(F.col("total_actual"), F.lit(1e-9)))
    .withColumn("acc_diq",   F.lit(1.0) - F.col("wape_diq"))
    .withColumn("acc_lego",  F.lit(1.0) - F.col("wape_lego"))
    .withColumn("acc_delta", F.col("acc_diq") - F.col("acc_lego"))
    .withColumn("bias_diq",  (F.col("total_actual") - F.col("total_diq")) / F.greatest(F.col("total_diq"), F.lit(1e-9)))
    .withColumn("bias_lego", (F.col("total_actual") - F.col("total_lego")) / F.greatest(F.col("total_lego"), F.lit(1e-9)))
    .orderBy("snapshot_week")
)

print("\nPer-snapshot accuracy:")
try:
    display(snap_stats)
except Exception:
    print(snap_stats.toPandas().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Per-category accuracy

# COMMAND ----------

cat_stats = (result
    .groupBy("category")
    .agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("cs_gtin").alias("n_gtins"),
        F.sum("actual").alias("total_actual"),
        F.sum("diq_predicted").alias("total_diq"),
        F.sum("lego_predicted").alias("total_lego"),
        F.sum("abs_error_diq").alias("abs_err_diq"),
        F.sum("abs_error_lego").alias("abs_err_lego"),
    )
    .withColumn("wape_diq",  F.col("abs_err_diq")  / F.greatest(F.col("total_actual"), F.lit(1e-9)))
    .withColumn("wape_lego", F.col("abs_err_lego") / F.greatest(F.col("total_actual"), F.lit(1e-9)))
    .withColumn("acc_diq",   F.lit(1.0) - F.col("wape_diq"))
    .withColumn("acc_lego",  F.lit(1.0) - F.col("wape_lego"))
    .withColumn("acc_delta", F.col("acc_diq") - F.col("acc_lego"))
    .withColumn("bias_diq",  (F.col("total_actual") - F.col("total_diq")) / F.greatest(F.col("total_diq"), F.lit(1e-9)))
    .withColumn("bias_lego", (F.col("total_actual") - F.col("total_lego")) / F.greatest(F.col("total_lego"), F.lit(1e-9)))
    .orderBy("category")
)

print("\nPer-category accuracy:")
try:
    display(cat_stats)
except Exception:
    print(cat_stats.toPandas().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Preview result

# COMMAND ----------

try:
    display(result.limit(50))
except Exception:
    print(result.limit(50).toPandas().to_string(index=False))


# COMMAND ----------

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC `result` DataFrame is ready. Columns for Excel pivot:
# MAGIC - **Dimensions**: `category`, `cs_gtin`, `year_week`, `snapshot_week`
# MAGIC - **Values**: `actual`, `diq_predicted`, `lego_predicted`, `abs_error_diq`, `abs_error_lego`
# MAGIC
# MAGIC To export: `result.toPandas().to_csv("/tmp/fr_accuracy.csv", index=False)`

# COMMAND ----------

# import pyspark.sql.functions as F

# PARENT_DIR        = "/Volumes/pds_feu_931272_dev/eu_france/transform/FR/DIQ_Test/lego_output"
# LEGO_PRED_COL     = "prediction_rf"
# KEY_COL           = "key"
# WEEK_COL          = "year_week"
# LIVE_SNAPSHOT     = "202616"
# EVAL_WEEK         = "202620"
# TARGET_GTIN       = "8711200845361"

# # def _exists(p):
# #     try:
# #         dbutils.fs.ls(p)
# #         return True
# #     except Exception:
# #         return False

# # def read_any(p):
# #     fmt = "delta" if _exists(f"{p}/_delta_log") else "parquet"
# #     return spark.read.format(fmt).load(p)

# # # --- Load DIQ inference (horizon=5) ---
# DIQ_PATH = f"{PARENT_DIR}/{LIVE_SNAPSHOT}/DIQ/TF_DIQ/fr_{LIVE_SNAPSHOT}_inference_forecast.parquet"
# print(DIQ_PATH)
# diq_sdf = spark.read.parquet(DIQ_PATH)
# diq_sdf = diq_sdf.filter(F.col("cs_gtin")== TARGET_GTIN)
# # diq_sdf = diq_sdf.withColumn(WEEK_COL, F.regexp_replace(F.col(WEEK_COL).cast("string"), "-", ""))
# diq_sdf = diq_sdf.filter(F.col("horizon") <= 13)
# diq_key = diq_sdf.select(
#     KEY_COL, WEEK_COL,
#     F.col("rf").alias("diq_rf")
# )


# # # --- Load LEGO TF (Period_Type=Forecast) ---
# LEGO_PATH = f"{PARENT_DIR}/{LIVE_SNAPSHOT}/TF"
# print(LEGO_PATH)
# lego_sdf = spark.read.format('delta').load(LEGO_PATH)
# # lego_sdf = read_any(LEGO_PATH)
# lego_sdf = lego_sdf.withColumn(WEEK_COL, F.regexp_replace(F.col(WEEK_COL).cast("string"), "-", ""))
# lego_sdf = lego_sdf.filter(F.col("Period_Type") == "Forecast")

# lego_key = lego_sdf.select(
#     KEY_COL, WEEK_COL,
#     F.col(LEGO_PRED_COL).alias("tf_prediction_rf"),
# )
# display(diq_key)

# # --- Full outer join at key × week, filter to eval week + GTIN ---
# joined = (diq_key
#     .join(lego_key, on=[KEY_COL, WEEK_COL], how="full")
#     .withColumn("cs_gtin", F.regexp_replace(F.col(KEY_COL).cast("string"), "_[^_]+$", ""))
#     .filter(F.col(WEEK_COL) == EVAL_WEEK)
#     .filter(F.col("cs_gtin") == TARGET_GTIN)
#     .withColumn("rf_diff", F.col("tf_prediction_rf") - F.col("diq_rf"))
#     .withColumn("in_diq", F.col("diq_rf").isNotNull())
#     .withColumn("in_tf", F.col("tf_prediction_rf").isNotNull())
#     .orderBy(KEY_COL)
# )

# # display(joined)

# COMMAND ----------

# # # import pandas as pd

# # # base_path = "/Volumes/pds_feu_931272_dev/data_science_team/uk_run"
# # # cat_list = ['CONDIMENT']
# # # cat_file_dict = {'CONDIMENTS' : 'CONDIMENT'}
# # # for cat, file_name in cat_file_dict.items():
# # #     data_path = f"{base_path}/{cat}/sourcedata/{file_name}_data.csv"
# # #     print(data_path)
# # #     df = pd.read_csv(data_path)
# # #     df['key'].nunique()

# # import pandas as pd
# # import os
# # import warnings
# # warnings.filterwarnings("ignore")

# # base_path = "/Volumes/pds_feu_931272_dev/data_science_team/uk_run"
# # cat_file_dict = {'CONDIMENTS': 'CONDIMENT'}

# # category_key_counts = {}

# # for cat, file_name in cat_file_dict.items():
# #     data_path = f"{base_path}/{cat}/sourcedata/{file_name}_data.csv"

# #     if not os.path.exists(data_path):
# #         print(f"Path not found for category '{cat}': {data_path}")
# #         continue

# #     print(data_path)
# #     df = pd.read_csv(data_path)
# #     category_key_counts[cat] = df['key'].nunique()
# import pyspark.sql.functions as F
# # print(category_key_counts)

# # df = spark.read.parquet("/Volumes/pds_feu_931272_dev/eu_uk/landing/FEU_LEGO/UK_PROD_NEW/baseline/snapshot/2026-03/DT/")

# distinct_count = (
#     df.filter(F.col('exclude_forecast') == 0)
#       .select(F.countDistinct('key'))
#       .collect()[0][0]
# )
# print(distinct_count)

# COMMAND ----------

