# Databricks notebook source
# MAGIC %md
# MAGIC # Create All 10 DACH Category Jobs
# MAGIC Run this notebook ONCE on your Databricks cluster.
# MAGIC It creates 10 independent jobs — one per category — all using the existing cluster.
# MAGIC After creation, go to Workflows → Jobs and click "Run Now" on all 10 to run in parallel.

# COMMAND ----------

import requests
import json

# Auto-detect Databricks host and token from notebook context
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Your existing cluster ID — get it from Compute → eda_stage_cluster_BATCH2 → JSON → cluster_id
# Or run: print(spark.conf.get("spark.databricks.clusterUsageTags.clusterId"))
CLUSTER_ID = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print(f"Using cluster: {CLUSTER_ID}")

# Your repo user (email)
REPO_USER = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
print(f"Repo user: {REPO_USER}")

REPO_PATH = f"/Workspace/Repos/{REPO_USER}/FEUagenticForecastingUpgrade"
print(f"Repo path: {REPO_PATH}")

# COMMAND ----------

CATEGORIES = {
    "SKIN_CARE":              {"short": "skincare",      "keys": 317},
    "ORAL_CARE":              {"short": "oral_care",     "keys": 221},
    "HAIR_CARE":              {"short": "haircare",      "keys": 126},
    "FABRIC_CLEANING":        {"short": "fab_clean",     "keys": 444},
    "HOME_HYGIENE":           {"short": "home_hygiene",  "keys": 882},
    "HEALTHY_SNACKING":       {"short": "healthy_snack", "keys": 1059},
    "DRESSINGS":              {"short": "dressings",     "keys": 1091},
    "SKIN_CLEANSING":         {"short": "skincleansing", "keys": 2375},
    "DEODORANTS_FRAGRANCES":  {"short": "deo",           "keys": 3215},
    "SCRATCH_COOKING_AIDS":   {"short": "cooking_aids",  "keys": 7818},
}

# COMMAND ----------

def create_job(cat_name, cat_info):
    """Create a Databricks job for one category."""
    config_file = f"{REPO_PATH}/config/config_de_{cat_info['short']}_databricks.yaml"

    job_spec = {
        "name": f"FEU-DACH-{cat_name}",
        "max_concurrent_runs": 1,
        "email_notifications": {
            "on_start": [REPO_USER],
            "on_success": [REPO_USER],
            "on_failure": [REPO_USER],
        },
        "tasks": [
            {
                "task_key": "full_pipeline",
                "description": f"Full pipeline: EDA → FA → Seg → FE → Train → Inference → Diag → Backtest ({cat_info['keys']} keys)",
                "existing_cluster_id": CLUSTER_ID,
                "spark_python_task": {
                    "python_file": f"{REPO_PATH}/databricks_runner.py",
                    "parameters": [
                        "--config", config_file,
                        "--stage", "full",
                    ],
                },
            }
        ],
        "tags": {
            "project": "feu-agentic-forecasting",
            "market": "DACH",
            "category": cat_name,
        },
    }

    response = requests.post(
        f"{db_host}/api/2.1/jobs/create",
        headers={"Authorization": f"Bearer {db_token}"},
        json=job_spec,
    )

    if response.status_code == 200:
        job_id = response.json()["job_id"]
        print(f"  ✓ Created: FEU-DACH-{cat_name} (job_id={job_id}, {cat_info['keys']} keys)")
        return job_id
    else:
        print(f"  ✗ FAILED: FEU-DACH-{cat_name} — {response.status_code}: {response.text}")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create All 10 Jobs

# COMMAND ----------

print("Creating 10 DACH category jobs...\n")

job_ids = {}
for cat_name, cat_info in CATEGORIES.items():
    job_id = create_job(cat_name, cat_info)
    if job_id:
        job_ids[cat_name] = job_id

print(f"\n{'='*60}")
print(f"Created {len(job_ids)}/10 jobs successfully")
print(f"{'='*60}")
for cat, jid in job_ids.items():
    print(f"  {cat}: job_id={jid}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Optional) Run All 10 Jobs NOW — in Parallel
# MAGIC Uncomment and run the cell below to trigger all 10 jobs simultaneously.

# COMMAND ----------

# UNCOMMENT TO RUN ALL 10 IN PARALLEL:

# print("Triggering all 10 jobs in parallel...\n")
# for cat_name, job_id in job_ids.items():
#     response = requests.post(
#         f"{db_host}/api/2.1/jobs/run-now",
#         headers={"Authorization": f"Bearer {db_token}"},
#         json={"job_id": job_id},
#     )
#     if response.status_code == 200:
#         run_id = response.json()["run_id"]
#         print(f"  ✓ Triggered: FEU-DACH-{cat_name} (run_id={run_id})")
#     else:
#         print(f"  ✗ FAILED: FEU-DACH-{cat_name} — {response.text}")
#
# print(f"\nAll jobs triggered! Monitor at: {db_host}/#/jobs")
