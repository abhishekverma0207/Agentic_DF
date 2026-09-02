#!/usr/bin/env python3
"""
Benchmark Comparison Script — reads pipeline output artifacts and compares
against PristineTVF benchmark.

PREREQUISITE: Run the full pipeline first:
    python runner.py --config config/config_de_skincare.yaml

Then run this:
    python compare_benchmark.py --category SKIN_CARE

This reads:
  - {artifact_base_path}/backtest_output/backtest_forecasts.csv (from runner.py)
  - TmpValidationData/DACH/CurrentForecastBenchmark.csv (benchmark)
  - sourcedata/DACH/{CATEGORY}.csv (for actuals)

Produces:
  - TmpValidationData/DACH/benchmark_reports/{CATEGORY}_summary.csv
  - TmpValidationData/DACH/benchmark_reports/{CATEGORY}_per_week.csv
  - TmpValidationData/DACH/benchmark_reports/{CATEGORY}_per_product.csv
  - TmpValidationData/DACH/benchmark_reports/{CATEGORY}_per_key.csv
  - TmpValidationData/DACH/benchmark_reports/{CATEGORY}_raw_forecasts.csv
  - TmpValidationData/DACH/benchmark_reports/{CATEGORY}_diagnostics.csv
"""
import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

CATEGORY_TO_BENCH = {
    'SKIN_CARE': 'SKIN CARE',
    'SKIN_CLEANSING': 'SKIN CLEANSING',
    'DEODORANTS_FRAGRANCES': 'DEODORANT & FRAGRANCE',
    'DRESSINGS': 'CONDIMENT',
    'FABRIC_CLEANING': 'FABRIC CLEANING',
    'HAIR_CARE': 'HAIR CARE',
    'HEALTHY_SNACKING': 'MINI MEAL',
    'HOME_HYGIENE': 'HOME & HYGIENE',
    'ORAL_CARE': 'ORAL CARE',
    'SCRATCH_COOKING_AIDS': 'COOKING AID',
}

CATEGORY_TO_CONFIG = {
    'SKIN_CARE': 'config/config_de_skincare.yaml',
    'SKIN_CLEANSING': 'config/config_de_skincleansing.yaml',
    'DEODORANTS_FRAGRANCES': 'config/config_de_deo.yaml',
    'DRESSINGS': 'config/config_de_dressings.yaml',
    'FABRIC_CLEANING': 'config/config_de_fab_clean.yaml',
    'HAIR_CARE': 'config/config_de_haircare.yaml',
    'HEALTHY_SNACKING': 'config/config_de_healthy_snack.yaml',
    'HOME_HYGIENE': 'config/config_de_home_hygiene.yaml',
    'ORAL_CARE': 'config/config_de_oral_care.yaml',
    'SCRATCH_COOKING_AIDS': 'config/config_de_cooking_aids.yaml',
}

BENCHMARK_PATH = "TmpValidationData/DACH/CurrentForecastBenchmark.csv"


def yw_numeric_to_dash(yw: int) -> str:
    return f"{yw // 100}-{yw % 100:02d}"

def compute_wape(a, p):
    t = np.sum(np.abs(a)); return float(np.sum(np.abs(a - p)) / t) if t > 1e-10 else np.nan

def compute_bias_pct(a, p):
    t = np.sum(np.abs(a)); return float((np.sum(p) - np.sum(a)) / t * 100) if t > 1e-10 else np.nan

def compute_mae(a, p):
    return float(np.mean(np.abs(a - p)))


def main():
    parser = argparse.ArgumentParser(description="Compare pipeline output vs PristineTVF benchmark")
    parser.add_argument('--category', required=True, choices=list(CATEGORY_TO_BENCH.keys()))
    parser.add_argument('--output-dir', default='TmpValidationData/DACH')
    args = parser.parse_args()

    category = args.category
    bench_category = CATEGORY_TO_BENCH[category]
    config_path = CATEGORY_TO_CONFIG[category]

    from config.schema import load_config_from_yaml
    cfg = load_config_from_yaml(config_path)
    KEY_COL = cfg.prediction_key_cols[0]
    DATE_COL = cfg.timestamp_col
    TARGET_COL = cfg.target_col
    artifact_dir = cfg.artifact_base_path

    t0 = time.time()
    logger.info("=" * 90)
    logger.info(f"BENCHMARK COMPARISON: {category}")
    logger.info("=" * 90)

    # =========================================================================
    # 1. LOAD PIPELINE BACKTEST OUTPUT
    # =========================================================================
    bt_path = os.path.join(artifact_dir, "backtest_output", "backtest_forecasts.csv")
    if not os.path.exists(bt_path):
        logger.error(f"Backtest output not found: {bt_path}")
        logger.error(f"Run the full pipeline first: python runner.py --config {config_path}")
        sys.exit(1)

    logger.info(f"\n[1/4] Loading pipeline backtest output: {bt_path}")
    our_df = pd.read_csv(bt_path, low_memory=False)
    from utils.period_utils import normalise_period_column
    our_df = normalise_period_column(our_df, DATE_COL)

    # Identify predicted and actual columns
    pred_col = 'predicted' if 'predicted' in our_df.columns else 'forecast'
    actual_col = TARGET_COL if TARGET_COL in our_df.columns else 'actual'

    logger.info(f"  Rows: {len(our_df):,}, Keys: {our_df[KEY_COL].nunique()}")
    logger.info(f"  Weeks: {sorted(our_df[DATE_COL].unique())[:5]}...{sorted(our_df[DATE_COL].unique())[-3:]}")
    logger.info(f"  Columns: {list(our_df.columns)}")

    # Filter to lag 4 if lag column exists
    if 'lag' in our_df.columns:
        our_df = our_df[our_df['lag'] == 4]
        logger.info(f"  Filtered to lag 4: {len(our_df):,} rows")

    our_df = our_df.rename(columns={pred_col: 'our_forecast', actual_col: TARGET_COL})
    our_df['our_forecast'] = pd.to_numeric(our_df['our_forecast'], errors='coerce').fillna(0)
    our_df[TARGET_COL] = pd.to_numeric(our_df[TARGET_COL], errors='coerce').fillna(0)

    # Also try loading inference forecast if available
    inf_path = os.path.join(artifact_dir, "model_artifacts", "inference_forecast.csv")
    has_inference = os.path.exists(inf_path)
    if has_inference:
        logger.info(f"  Also found inference forecast: {inf_path}")

    # =========================================================================
    # 2. LOAD BENCHMARK
    # =========================================================================
    logger.info(f"\n[2/4] Loading PristineTVF benchmark (Category={bench_category}, InScope=1)...")
    bench = pd.read_csv(BENCHMARK_PATH, low_memory=False)
    bench = bench[(bench['Category'] == bench_category) & (bench['InScopeFlag'] == 1)].copy()
    bench['PristineTVF'] = pd.to_numeric(bench['PristineTVF'], errors='coerce').fillna(0)
    bench['key'] = bench['ProductCode'].astype(str) + '_' + bench['CustomerCode'].astype(str).str.zfill(10)
    bench['week_dash'] = bench['YearWeek'].apply(yw_numeric_to_dash)
    bench_agg = bench.groupby(['key', 'week_dash']).agg(
        benchmark_forecast=('PristineTVF', 'sum'),
    ).reset_index().rename(columns={'key': KEY_COL, 'week_dash': DATE_COL})
    logger.info(f"  Benchmark rows: {len(bench_agg):,}, keys: {bench_agg[KEY_COL].nunique()}")

    # =========================================================================
    # 3. MERGE ON COMMON KEYS AND WEEKS
    # =========================================================================
    logger.info(f"\n[3/4] Merging on common keys and weeks...")
    # If our data doesn't have actuals (some backtest formats), load from source
    if TARGET_COL not in our_df.columns or our_df[TARGET_COL].isna().all():
        source_df = pd.read_csv(f"sourcedata/DACH/{category}.csv", low_memory=False)
        source_df = normalise_period_column(source_df, DATE_COL)
        actuals = source_df[[KEY_COL, DATE_COL, TARGET_COL]].drop_duplicates()
        our_df = our_df.merge(actuals, on=[KEY_COL, DATE_COL], how='left', suffixes=('_drop', ''))
        if f'{TARGET_COL}_drop' in our_df.columns:
            our_df.drop(columns=[f'{TARGET_COL}_drop'], inplace=True)
        our_df[TARGET_COL] = our_df[TARGET_COL].fillna(0)

    merged = our_df[[KEY_COL, DATE_COL, 'our_forecast', TARGET_COL]].merge(
        bench_agg, on=[KEY_COL, DATE_COL], how='inner'
    )
    merged['ProductCode'] = merged[KEY_COL].str.split('_').str[0]
    merged['CustomerCode'] = merged[KEY_COL].str.split('_').str[1]

    our_keys = set(our_df[KEY_COL].unique())
    bench_keys = set(bench_agg[KEY_COL].unique())
    logger.info(f"  Our keys: {len(our_keys)}, Benchmark keys: {len(bench_keys)}")
    logger.info(f"  Common keys: {len(our_keys & bench_keys)}")
    logger.info(f"  Common key-weeks: {len(merged):,}")

    if len(merged) == 0:
        logger.error("No common key-weeks found! Check date formats and key alignment.")
        sys.exit(1)

    # =========================================================================
    # 4. BUILD CSV REPORT
    # =========================================================================
    logger.info(f"\n[4/4] Building comparison report...")
    report_prefix = os.path.join(args.output_dir, f'{category}')
    os.makedirs(args.output_dir, exist_ok=True)

    actual = merged[TARGET_COL].values
    our_pred = merged['our_forecast'].values
    bench_pred = merged['benchmark_forecast'].values

    # Product-level aggregation
    prod_agg = merged.groupby(['ProductCode', DATE_COL]).agg(
        actual=(TARGET_COL, 'sum'), our_fc=('our_forecast', 'sum'), bench_fc=('benchmark_forecast', 'sum'),
    ).reset_index()
    pa, po, pb = prod_agg['actual'].values, prod_agg['our_fc'].values, prod_agg['bench_fc'].values

    # ── CSV 1: Summary ──────────────────────────────────────────
    summary = pd.DataFrame({
        'Metric': [
            'Category', 'Benchmark Category', 'Run Date',
            'Common Keys', 'Common Products', 'Backtest Weeks',
            '', '--- KEY × WEEK ---',
            'Our WAPE', 'Benchmark WAPE', 'WAPE Difference', 'Our Bias%', 'Benchmark Bias%',
            'Total Actual', 'Total Our Forecast', 'Total Benchmark',
            '', '--- PRODUCT × WEEK ---',
            'Our WAPE', 'Benchmark WAPE', 'WAPE Difference', 'Our Bias%', 'Benchmark Bias%',
        ],
        'Value': [
            category, bench_category, datetime.now().strftime('%Y-%m-%d %H:%M'),
            merged[KEY_COL].nunique(), merged['ProductCode'].nunique(),
            merged[DATE_COL].nunique(),
            '', '',
            f"{compute_wape(actual, our_pred):.1%}", f"{compute_wape(actual, bench_pred):.1%}",
            f"{compute_wape(actual, our_pred)-compute_wape(actual, bench_pred):+.1%}",
            f"{compute_bias_pct(actual, our_pred):+.1f}%", f"{compute_bias_pct(actual, bench_pred):+.1f}%",
            f"{actual.sum():,.0f}", f"{our_pred.sum():,.0f}", f"{bench_pred.sum():,.0f}",
            '', '',
            f"{compute_wape(pa, po):.1%}", f"{compute_wape(pa, pb):.1%}",
            f"{compute_wape(pa, po)-compute_wape(pa, pb):+.1%}",
            f"{compute_bias_pct(pa, po):+.1f}%", f"{compute_bias_pct(pa, pb):+.1f}%",
        ],
    })
    summary.to_csv(f"{report_prefix}_summary.csv", index=False)

    # ── CSV 2: Per-Week ──────────────────────────────────────────
    weeks = []
    for w in sorted(merged[DATE_COL].unique()):
        wk = merged[merged[DATE_COL] == w]
        a, o, b = wk[TARGET_COL].values, wk['our_forecast'].values, wk['benchmark_forecast'].values
        weeks.append({
            'Week': w, 'N_Keys': len(wk), 'Total_Actual': a.sum(),
            'Total_Ours': o.sum(), 'Total_Benchmark': b.sum(),
            'Our_WAPE': compute_wape(a, o), 'Benchmark_WAPE': compute_wape(a, b),
            'WAPE_Gap': compute_wape(a, o) - compute_wape(a, b),
            'Our_Bias%': compute_bias_pct(a, o), 'Bench_Bias%': compute_bias_pct(a, b),
            'Winner': 'OURS' if compute_wape(a, o) < compute_wape(a, b) else 'BENCHMARK',
        })
    pd.DataFrame(weeks).to_csv(f"{report_prefix}_per_week.csv", index=False)

    # ── CSV 3: Per-Product ───────────────────────────────────────
    prods = []
    for prod in merged['ProductCode'].unique():
        pk = merged[merged['ProductCode'] == prod]
        a, o, b = pk[TARGET_COL].values, pk['our_forecast'].values, pk['benchmark_forecast'].values
        prods.append({
            'ProductCode': prod, 'N_Keys': pk[KEY_COL].nunique(),
            'N_Weeks': pk[DATE_COL].nunique(), 'Total_Actual': a.sum(),
            'Total_Ours': o.sum(), 'Total_Benchmark': b.sum(),
            'Our_WAPE': compute_wape(a, o), 'Benchmark_WAPE': compute_wape(a, b),
            'WAPE_Gap': compute_wape(a, o) - compute_wape(a, b),
            'Our_Bias%': compute_bias_pct(a, o), 'Bench_Bias%': compute_bias_pct(a, b),
            'Winner': 'OURS' if compute_wape(a, o) < compute_wape(a, b) else 'BENCHMARK',
        })
    pd.DataFrame(prods).sort_values('Total_Actual', ascending=False).to_csv(
        f"{report_prefix}_per_product.csv", index=False)

    # ── CSV 4: Per-Key ───────────────────────────────────────────
    keys = []
    for key in merged[KEY_COL].unique():
        kk = merged[merged[KEY_COL] == key]
        a, o, b = kk[TARGET_COL].values, kk['our_forecast'].values, kk['benchmark_forecast'].values
        keys.append({
            'Key': key, 'ProductCode': key.split('_')[0], 'CustomerCode': key.split('_')[1],
            'N_Weeks': len(kk), 'Total_Actual': a.sum(),
            'Total_Ours': o.sum(), 'Total_Benchmark': b.sum(),
            'Our_WAPE': compute_wape(a, o), 'Benchmark_WAPE': compute_wape(a, b),
            'WAPE_Gap': compute_wape(a, o) - compute_wape(a, b),
            'Our_Bias%': compute_bias_pct(a, o), 'Bench_Bias%': compute_bias_pct(a, b),
            'Zero_Fraction': (a == 0).mean(),
            'Winner': 'OURS' if compute_wape(a, o) < compute_wape(a, b) else 'BENCHMARK',
        })
    pd.DataFrame(keys).sort_values('Total_Actual', ascending=False).to_csv(
        f"{report_prefix}_per_key.csv", index=False)

    # ── CSV 5: Raw Forecasts ─────────────────────────────────────
    merged.to_csv(f"{report_prefix}_raw_forecasts.csv", index=False)

    # ── CSV 6: Diagnostics ───────────────────────────────────────
    zero_mask = merged[TARGET_COL] == 0
    nz_mask = ~zero_mask
    diag = pd.DataFrame({
        'Metric': [
            'Total rows', 'Zero-demand rows', 'Zero-demand %',
            'Our over-forecast rate', 'Benchmark over-forecast rate',
            'Our avg forecast on zeros', 'Benchmark avg forecast on zeros',
            'WAPE on non-zero only (ours)', 'WAPE on non-zero only (benchmark)',
        ],
        'Value': [
            len(merged), zero_mask.sum(), f"{zero_mask.mean():.1%}",
            f"{(our_pred > actual).mean():.1%}", f"{(bench_pred > actual).mean():.1%}",
            f"{our_pred[zero_mask].mean():.1f}" if zero_mask.any() else 'N/A',
            f"{bench_pred[zero_mask].mean():.1f}" if zero_mask.any() else 'N/A',
            f"{compute_wape(actual[nz_mask], our_pred[nz_mask]):.1%}" if nz_mask.any() else 'N/A',
            f"{compute_wape(actual[nz_mask], bench_pred[nz_mask]):.1%}" if nz_mask.any() else 'N/A',
        ],
    })
    diag.to_csv(f"{report_prefix}_diagnostics.csv", index=False)

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================
    total = time.time() - t0
    our_w = compute_wape(actual, our_pred)
    bench_w = compute_wape(actual, bench_pred)
    prod_our_w = compute_wape(pa, po)
    prod_bench_w = compute_wape(pa, pb)

    logger.info(f"\n{'='*90}")
    logger.info(f"BENCHMARK COMPARISON COMPLETE — {category}")
    logger.info(f"{'='*90}")
    logger.info(f"  KEY × WEEK:   Ours={our_w:.1%}  Bench={bench_w:.1%}  ({our_w-bench_w:+.1%})")
    logger.info(f"  PROD × WEEK:  Ours={prod_our_w:.1%}  Bench={prod_bench_w:.1%}  ({prod_our_w-prod_bench_w:+.1%})")
    logger.info(f"  Bias:         Ours={compute_bias_pct(actual, our_pred):+.1f}%  Bench={compute_bias_pct(actual, bench_pred):+.1f}%")
    logger.info(f"  Common keys:  {merged[KEY_COL].nunique()}, Products: {merged['ProductCode'].nunique()}")
    weeks_won = sum(1 for w in weeks if w['Winner'] == 'OURS')
    logger.info(f"  Weeks won:    {weeks_won}/{len(weeks)}")
    prods_won = sum(1 for p in prods if p['Winner'] == 'OURS')
    logger.info(f"  Products won: {prods_won}/{len(prods)}")
    logger.info(f"  Reports:      {report_prefix}_*.csv")
    logger.info(f"  Time:         {total:.1f}s")
    logger.info(f"{'='*90}")


if __name__ == '__main__':
    main()
