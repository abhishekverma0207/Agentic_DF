# Databricks notebook source
# MAGIC %md
# MAGIC # DACH Benchmark Comparison: Agentic Forecast vs PristineTVF
# MAGIC
# MAGIC Run this AFTER all category backtesting jobs have completed.
# MAGIC Reads backtest_forecasts.csv from each category's artifacts and compares against PristineTVF benchmark.

# COMMAND ----------

import pandas as pd
import numpy as np
import os

# Paths
VOLUMES_BASE = "/Volumes/pds_feu_931272_dev/data_science_team/dach_rerun"
BENCHMARK_PATH = f"{VOLUMES_BASE}/LEGO_BENCHMARK/CurrentForecastBenchmark.csv"

# Category mapping: our folder name → benchmark Category column
CATEGORY_MAP = {
    "SKIN_CARE": "SKIN CARE",
    "SKIN_CLEANSING": "SKIN CLEANSING",
    "DEODORANTS_FRAGRANCES": "DEODORANT & FRAGRANCE",
    "DRESSINGS": "CONDIMENT",
    "FABRIC_CLEANING": "FABRIC CLEANING",
    "HAIR_CARE": "HAIR CARE",
    "HEALTHY_SNACKING": "MINI MEAL",
    "HOME_HYGIENE": "HOME & HYGIENE",
    "ORAL_CARE": "ORAL CARE",
    "SCRATCH_COOKING_AIDS": "COOKING AID",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Benchmark

# COMMAND ----------

print(f"Loading benchmark from: {BENCHMARK_PATH}")
bench_raw = pd.read_csv(BENCHMARK_PATH, low_memory=False)
print(f"Benchmark: {len(bench_raw):,} rows, {bench_raw['Category'].nunique()} categories")
print(f"Categories: {sorted(bench_raw['Category'].unique())}")
print(f"InScopeFlag distribution: {bench_raw['InScopeFlag'].value_counts().to_dict()}")
print(f"WeeklagIndicator: {sorted(bench_raw['WeeklagIndicator'].unique())}")

# Filter to InScope only
bench = bench_raw[bench_raw['InScopeFlag'] == 1].copy()
bench['PristineTVF'] = pd.to_numeric(bench['PristineTVF'], errors='coerce').fillna(0)
bench['ActualShipment'] = pd.to_numeric(bench['ActualShipment'], errors='coerce').fillna(0)

# Build key: ProductCode_CustomerCode (zero-pad customer to 10 digits to match our keys)
bench['key'] = bench['ProductCode'].astype(str) + '_' + bench['CustomerCode'].astype(str).str.zfill(10)

# Convert YearWeek (YYYYWW) to YYYY-WW (dash format) to match our backtest output
bench['week_dash'] = bench['YearWeek'].apply(lambda yw: f"{yw // 100}-{yw % 100:02d}")

# Sum PristineTVF by key × week (aggregate multiple promo rows per key-week)
bench_agg = bench.groupby(['key', 'week_dash', 'Category']).agg(
    benchmark_forecast=('PristineTVF', 'sum'),
).reset_index()

print(f"\nBenchmark (InScope=1, aggregated): {len(bench_agg):,} key-weeks")
print(f"Unique keys: {bench_agg['key'].nunique()}")
print(f"Weeks: {sorted(bench_agg['week_dash'].unique())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Our Backtest Forecasts (All Categories)

# COMMAND ----------

def compute_wape(actual, predicted):
    total = np.sum(np.abs(actual))
    return float(np.sum(np.abs(actual - predicted)) / total) if total > 1e-10 else np.nan

def compute_bias_pct(actual, predicted):
    total = np.sum(np.abs(actual))
    return float((np.sum(predicted) - np.sum(actual)) / total * 100) if total > 1e-10 else np.nan

all_our_forecasts = []
categories_loaded = []

for our_cat, bench_cat in CATEGORY_MAP.items():
    bt_path = f"{VOLUMES_BASE}/{our_cat}/artifacts/backtest_output/backtest_forecasts.csv"

    if not os.path.exists(bt_path):
        print(f"  ✗ {our_cat}: backtest not found at {bt_path}")
        continue

    bt = pd.read_csv(bt_path, low_memory=False)

    # Identify columns
    key_col = 'key' if 'key' in bt.columns else 'Model_Hierarchy'
    date_col = 'Shipment_week' if 'Shipment_week' in bt.columns else bt.columns[1]
    pred_col = 'predicted' if 'predicted' in bt.columns else 'forecast'
    actual_col = 'actual' if 'actual' in bt.columns else 'Actuals'

    # Filter to lag 4 if lag column exists
    # Lag 0 = first week after history cutoff, Lag 4 = 5th week
    if 'lag' in bt.columns:
        bt_lag4 = bt[bt['lag'] == 4].copy()
        if len(bt_lag4) == 0:
            # Try forecast_step = 5 (1-indexed) or lag = 4
            bt_lag4 = bt[bt.get('forecast_step', bt.get('lag', pd.Series())) == 5].copy()
        if len(bt_lag4) == 0:
            bt_lag4 = bt.copy()  # Use all if no lag 4
            print(f"  ⚠ {our_cat}: no lag 4 found, using all {len(bt)} rows")
    else:
        bt_lag4 = bt.copy()

    bt_lag4 = bt_lag4.rename(columns={
        key_col: 'key', date_col: 'week', pred_col: 'our_forecast', actual_col: 'actual'
    })
    bt_lag4['our_forecast'] = pd.to_numeric(bt_lag4['our_forecast'], errors='coerce').fillna(0)
    bt_lag4['actual'] = pd.to_numeric(bt_lag4['actual'], errors='coerce').fillna(0)
    bt_lag4['week'] = bt_lag4['week'].astype(str)
    bt_lag4['our_category'] = our_cat
    bt_lag4['bench_category'] = bench_cat

    all_our_forecasts.append(bt_lag4[['key', 'week', 'our_forecast', 'actual', 'our_category', 'bench_category']])
    categories_loaded.append(our_cat)
    print(f"  ✓ {our_cat}: {len(bt_lag4):,} rows, {bt_lag4['key'].nunique()} keys")

if not all_our_forecasts:
    raise ValueError("No backtest forecasts found for any category!")

our_df = pd.concat(all_our_forecasts, ignore_index=True)
print(f"\nTotal our forecasts: {len(our_df):,} rows across {len(categories_loaded)} categories")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge & Compare

# COMMAND ----------

# Merge our forecasts with benchmark on (key, week)
merged = our_df.merge(
    bench_agg[['key', 'week_dash', 'benchmark_forecast']],
    left_on=['key', 'week'],
    right_on=['key', 'week_dash'],
    how='inner',
)

print(f"Merged: {len(merged):,} common key-weeks")
print(f"Common keys: {merged['key'].nunique()}")
print(f"Our keys not in benchmark: {our_df['key'].nunique() - merged['key'].nunique()}")
print(f"Benchmark keys not in ours: {bench_agg['key'].nunique() - merged['key'].nunique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results by Category

# COMMAND ----------

print(f"\n{'='*100}")
print(f"BENCHMARK COMPARISON: Key × Week Level")
print(f"{'='*100}")
print(f"\n{'Category':<30} {'Keys':>6} {'Weeks':>6} {'Our WAPE':>10} {'Bench WAPE':>12} {'Gap':>8} {'Our Bias%':>10} {'Bench Bias%':>12} {'Winner':>8}")
print(f"{'-'*100}")

summary_rows = []
total_our_win = 0
total_cats = 0

for our_cat in sorted(categories_loaded):
    cat_df = merged[merged['our_category'] == our_cat]
    if len(cat_df) == 0:
        continue

    a = cat_df['actual'].values
    o = cat_df['our_forecast'].values
    b = cat_df['benchmark_forecast'].values

    our_w = compute_wape(a, o)
    bench_w = compute_wape(a, b)
    our_b = compute_bias_pct(a, o)
    bench_b = compute_bias_pct(a, b)
    gap = our_w - bench_w
    winner = "OURS" if our_w < bench_w else "BENCH"

    if our_w < bench_w:
        total_our_win += 1
    total_cats += 1

    print(f"{our_cat:<30} {cat_df['key'].nunique():>6} {cat_df['week'].nunique():>6} "
          f"{our_w:>9.1%} {bench_w:>11.1%} {gap:>+7.1%} "
          f"{our_b:>9.1f}% {bench_b:>11.1f}% {winner:>8}")

    summary_rows.append({
        'Category': our_cat,
        'Benchmark_Category': CATEGORY_MAP[our_cat],
        'Common_Keys': cat_df['key'].nunique(),
        'Common_Weeks': cat_df['week'].nunique(),
        'Our_WAPE': our_w,
        'Benchmark_WAPE': bench_w,
        'WAPE_Gap': gap,
        'Our_Bias_Pct': our_b,
        'Benchmark_Bias_Pct': bench_b,
        'Winner': winner,
        'Total_Actual': float(a.sum()),
        'Total_Our_Forecast': float(o.sum()),
        'Total_Benchmark': float(b.sum()),
    })

# Overall
a_all = merged['actual'].values
o_all = merged['our_forecast'].values
b_all = merged['benchmark_forecast'].values

print(f"{'-'*100}")
print(f"{'OVERALL':<30} {merged['key'].nunique():>6} {merged['week'].nunique():>6} "
      f"{compute_wape(a_all, o_all):>9.1%} {compute_wape(a_all, b_all):>11.1%} "
      f"{compute_wape(a_all, o_all)-compute_wape(a_all, b_all):>+7.1%} "
      f"{compute_bias_pct(a_all, o_all):>9.1f}% {compute_bias_pct(a_all, b_all):>11.1f}% "
      f"{'OURS' if compute_wape(a_all, o_all) < compute_wape(a_all, b_all) else 'BENCH':>8}")
print(f"\nWe win {total_our_win}/{total_cats} categories")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results by Category — Product × Week Level

# COMMAND ----------

merged['ProductCode'] = merged['key'].str.split('_').str[0]

print(f"\n{'='*100}")
print(f"BENCHMARK COMPARISON: Product × Week Level (aggregated over customers)")
print(f"{'='*100}")
print(f"\n{'Category':<30} {'Products':>9} {'Our WAPE':>10} {'Bench WAPE':>12} {'Gap':>8} {'Winner':>8}")
print(f"{'-'*80}")

for our_cat in sorted(categories_loaded):
    cat_df = merged[merged['our_category'] == our_cat]
    if len(cat_df) == 0:
        continue

    prod_agg = cat_df.groupby(['ProductCode', 'week']).agg(
        actual=('actual', 'sum'),
        our_fc=('our_forecast', 'sum'),
        bench_fc=('benchmark_forecast', 'sum'),
    ).reset_index()

    a = prod_agg['actual'].values
    o = prod_agg['our_fc'].values
    b = prod_agg['bench_fc'].values

    our_w = compute_wape(a, o)
    bench_w = compute_wape(a, b)
    winner = "OURS" if our_w < bench_w else "BENCH"

    print(f"{our_cat:<30} {prod_agg['ProductCode'].nunique():>9} "
          f"{our_w:>9.1%} {bench_w:>11.1%} {our_w-bench_w:>+7.1%} {winner:>8}")

# Overall product level
prod_all = merged.groupby(['ProductCode', 'week']).agg(
    actual=('actual', 'sum'), our_fc=('our_forecast', 'sum'), bench_fc=('benchmark_forecast', 'sum'),
).reset_index()
print(f"{'-'*80}")
print(f"{'OVERALL':<30} {prod_all['ProductCode'].nunique():>9} "
      f"{compute_wape(prod_all['actual'].values, prod_all['our_fc'].values):>9.1%} "
      f"{compute_wape(prod_all['actual'].values, prod_all['bench_fc'].values):>11.1%} "
      f"{compute_wape(prod_all['actual'].values, prod_all['our_fc'].values)-compute_wape(prod_all['actual'].values, prod_all['bench_fc'].values):>+7.1%} "
      f"{'OURS' if compute_wape(prod_all['actual'].values, prod_all['our_fc'].values) < compute_wape(prod_all['actual'].values, prod_all['bench_fc'].values) else 'BENCH':>8}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Summary to CSV

# COMMAND ----------

summary_df = pd.DataFrame(summary_rows)
output_base = f"{VOLUMES_BASE}/LEGO_BENCHMARK/benchmark_comparison_results"

# Summary by category
summary_df.to_csv(f"{output_base}_by_category.csv", index=False)

# Raw merged data
merged.to_csv(f"{output_base}_raw_key_week.csv", index=False)

# Product level
prod_all_detail = merged.groupby(['our_category', 'ProductCode', 'week']).agg(
    actual=('actual', 'sum'), our_forecast=('our_forecast', 'sum'), benchmark_forecast=('benchmark_forecast', 'sum'),
).reset_index()
prod_all_detail.to_csv(f"{output_base}_product_week.csv", index=False)

print(f"\nSaved to: {output_base}_*.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diagnostic: Per-Week Breakdown (All Categories)

# COMMAND ----------

print(f"\n{'='*80}")
print(f"PER-WEEK BREAKDOWN (All Categories Combined)")
print(f"{'='*80}")
print(f"\n{'Week':<12} {'Actual':>12} {'Ours':>12} {'Bench':>12} {'Our WAPE':>10} {'Bench WAPE':>12} {'Winner':>8}")
print(f"{'-'*80}")

for week in sorted(merged['week'].unique()):
    wk = merged[merged['week'] == week]
    a = wk['actual'].values
    o = wk['our_forecast'].values
    b = wk['benchmark_forecast'].values
    our_w = compute_wape(a, o)
    bench_w = compute_wape(a, b)
    winner = "OURS" if our_w < bench_w else "BENCH"
    print(f"{week:<12} {a.sum():>12,.0f} {o.sum():>12,.0f} {b.sum():>12,.0f} "
          f"{our_w:>9.1%} {bench_w:>11.1%} {winner:>8}")
