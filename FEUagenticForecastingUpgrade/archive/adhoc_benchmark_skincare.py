#!/usr/bin/env python3
"""
Ad-hoc Benchmark Comparison: Our Agentic Forecast vs PristineTVF Benchmark
Category: DACH SKIN CARE

Workflow:
1. Load our SKIN_CARE data, train models, generate rolling-origin lag-4 forecasts
   for weeks 2025-52 to 2026-10
2. Load PristineTVF benchmark (InScopeFlag=1), sum by ProductCode x CustomerCode x YearWeek
3. Merge on common keys, compare accuracy & bias at:
   a) Key × Week level
   b) ProductCode × Week level (aggregated)
4. Investigate areas of improvement
5. Print a full comparison report

Usage:
    source .venv/bin/activate
    python adhoc_benchmark_skincare.py
"""
import warnings
warnings.filterwarnings('ignore')

import logging
import os
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ─── CONFIG ─────────────────────────────────────────────────────────────────
DATA_PATH = "sourcedata/DACH/SKIN_CARE.csv"
BENCHMARK_PATH = "TmpValidationData/DACH/CurrentForecastBenchmark.csv"
KEY_COL = "Model_Hierarchy"
DATE_COL = "Shipment_week"
TARGET_COL = "Actuals"
FORECAST_LAG = 4

# Target backtest period: weeks where benchmark has InScope=1 forecasts
# These are weeks 2025-52 through 2026-10 in our YYYY-WW format
# The benchmark uses YYYYWW (202552..202610)
TARGET_WEEKS_START = "2025-52"
TARGET_WEEKS_END = "2026-10"


def yw_dash_to_numeric(yw: str) -> int:
    """Convert 'YYYY-WW' → YYYYWW integer. E.g., '2025-52' → 202552."""
    parts = str(yw).split('-')
    return int(parts[0]) * 100 + int(parts[1])


def yw_numeric_to_dash(yw: int) -> str:
    """Convert YYYYWW → 'YYYY-WW'. E.g., 202552 → '2025-52'."""
    year = yw // 100
    week = yw % 100
    return f"{year}-{week:02d}"


def compute_wape(actual, predicted):
    """WAPE = sum(|actual - predicted|) / sum(|actual|)."""
    total_actual = np.sum(np.abs(actual))
    if total_actual < 1e-10:
        return np.nan
    return float(np.sum(np.abs(actual - predicted)) / total_actual)


def compute_bias(actual, predicted):
    """Bias = (sum(predicted) - sum(actual)) / sum(actual) * 100%."""
    total_actual = np.sum(np.abs(actual))
    if total_actual < 1e-10:
        return np.nan
    return float((np.sum(predicted) - np.sum(actual)) / total_actual * 100)


def main():
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("BENCHMARK COMPARISON: Agentic Forecast vs PristineTVF — DACH SKIN CARE")
    logger.info("=" * 80)

    # =========================================================================
    # STEP 1: LOAD AND PREPARE OUR DATA
    # =========================================================================
    logger.info("\n[STEP 1] Loading our SKIN_CARE data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    from utils.period_utils import normalise_period_column
    df = normalise_period_column(df, DATE_COL)

    # History cutoff
    period_totals = df.groupby(DATE_COL)[TARGET_COL].sum()
    positive = period_totals[period_totals > 0]
    history_cutoff = str(positive.index.max())
    logger.info(f"  Rows: {len(df):,}, Keys: {df[KEY_COL].nunique()}, History cutoff: {history_cutoff}")

    # Define target weeks for backtesting
    all_weeks = sorted(df[DATE_COL].unique())
    target_weeks = [w for w in all_weeks if w >= TARGET_WEEKS_START and w <= TARGET_WEEKS_END]
    logger.info(f"  Target backtest weeks: {target_weeks}")

    # =========================================================================
    # STEP 2: TRAIN MODELS AND GENERATE LAG-4 FORECASTS
    # =========================================================================
    logger.info("\n[STEP 2] Training models and generating lag-4 forecasts...")
    logger.info("  Using rolling-origin backtesting with direct lag-4 models")

    # Prepare features
    hist_df = df[df[DATE_COL] <= history_cutoff].copy()
    hist_df.sort_values([KEY_COL, DATE_COL], inplace=True)

    for lag in [1, 2, 4, 8, 13]:
        hist_df[f'{TARGET_COL}_lag_{lag}'] = hist_df.groupby(KEY_COL)[TARGET_COL].shift(lag)
    for w in [4, 13]:
        hist_df[f'{TARGET_COL}_roll_mean_{w}'] = hist_df.groupby(KEY_COL)[TARGET_COL].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )

    feature_cols = [c for c in hist_df.columns
                    if c.startswith(f'{TARGET_COL}_lag_') or c.startswith(f'{TARGET_COL}_roll_')]

    # For each target week, we need to train on data up to (week - lag4) and predict
    # Rolling origin: for week W at lag 4, train on all data up to W-4, predict W
    import lightgbm as lgb

    all_forecasts = []
    origin_count = 0

    for target_week in target_weeks:
        # Find the week that is 4 weeks before target_week
        tw_idx = all_weeks.index(target_week) if target_week in all_weeks else -1
        if tw_idx < FORECAST_LAG:
            continue
        origin_week = all_weeks[tw_idx - FORECAST_LAG]

        # Train on everything up to origin_week
        train_mask = hist_df[DATE_COL] <= origin_week
        val_mask = (hist_df[DATE_COL] > origin_week) & (hist_df[DATE_COL] <= target_week)

        train_data = hist_df[train_mask].dropna(subset=feature_cols)
        if len(train_data) < 100:
            continue

        X_train = train_data[feature_cols].fillna(0).values
        y_train = train_data[TARGET_COL].fillna(0).values

        # Train LightGBM
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, verbosity=-1,
        )
        model.fit(X_train, y_train)

        # Get features at origin_week for each key, predict target_week
        origin_data = hist_df[hist_df[DATE_COL] == origin_week]
        if len(origin_data) == 0:
            continue

        X_pred = origin_data[feature_cols].fillna(0).values
        preds = np.maximum(model.predict(X_pred), 0)

        # Collect forecasts
        for i, (_, row) in enumerate(origin_data.iterrows()):
            all_forecasts.append({
                KEY_COL: row[KEY_COL],
                DATE_COL: target_week,
                'our_forecast': preds[i],
            })
        origin_count += 1

    our_fc_df = pd.DataFrame(all_forecasts)
    logger.info(f"  Generated {len(our_fc_df):,} forecasts across {origin_count} origins")
    logger.info(f"  Forecast weeks: {sorted(our_fc_df[DATE_COL].unique())}")

    # Merge actuals from our data
    actuals = hist_df[[KEY_COL, DATE_COL, TARGET_COL]].copy()
    our_fc_df = our_fc_df.merge(actuals, on=[KEY_COL, DATE_COL], how='left')
    our_fc_df[TARGET_COL] = our_fc_df[TARGET_COL].fillna(0)

    # =========================================================================
    # STEP 3: LOAD AND PREPARE BENCHMARK DATA
    # =========================================================================
    logger.info("\n[STEP 3] Loading PristineTVF benchmark...")
    bench = pd.read_csv(BENCHMARK_PATH, low_memory=False)

    # Filter: SKIN CARE, InScopeFlag=1
    bench = bench[(bench['Category'] == 'SKIN CARE') & (bench['InScopeFlag'] == 1)].copy()
    logger.info(f"  Benchmark rows (SKIN CARE, InScope=1): {len(bench):,}")

    # Convert PristineTVF to numeric
    bench['PristineTVF'] = pd.to_numeric(bench['PristineTVF'], errors='coerce').fillna(0)
    bench['ActualShipment'] = pd.to_numeric(bench['ActualShipment'], errors='coerce').fillna(0)

    # Build key: ProductCode_CustomerCode (zero-pad customer to 10 digits)
    bench['CustomerCode_padded'] = bench['CustomerCode'].astype(str).str.zfill(10)
    bench['key'] = bench['ProductCode'].astype(str) + '_' + bench['CustomerCode_padded']

    # Convert YearWeek to our YYYY-WW format
    bench['week_dash'] = bench['YearWeek'].apply(yw_numeric_to_dash)

    # Sum PristineTVF by key × week (aggregate multiple rows per key-week)
    bench_agg = bench.groupby(['key', 'week_dash']).agg(
        benchmark_forecast=('PristineTVF', 'sum'),
    ).reset_index()
    bench_agg.rename(columns={'key': KEY_COL, 'week_dash': DATE_COL}, inplace=True)

    # Filter to target weeks
    bench_agg = bench_agg[bench_agg[DATE_COL].isin(target_weeks)]
    logger.info(f"  Benchmark keys: {bench_agg[KEY_COL].nunique()}, weeks: {sorted(bench_agg[DATE_COL].unique())}")

    # =========================================================================
    # STEP 4: MERGE AND COMPARE — KEY × WEEK LEVEL
    # =========================================================================
    logger.info("\n[STEP 4] Merging forecasts on common keys...")

    merged = our_fc_df.merge(bench_agg, on=[KEY_COL, DATE_COL], how='inner')
    logger.info(f"  Common key-weeks: {len(merged):,}")
    logger.info(f"  Common keys: {merged[KEY_COL].nunique()}")
    logger.info(f"  Our keys: {our_fc_df[KEY_COL].nunique()}, Benchmark keys: {bench_agg[KEY_COL].nunique()}")

    # Only keys in benchmark but not in ours (and vice versa)
    our_keys = set(our_fc_df[KEY_COL].unique())
    bench_keys = set(bench_agg[KEY_COL].unique())
    common_keys = our_keys & bench_keys
    logger.info(f"  Keys only in ours: {len(our_keys - bench_keys)}")
    logger.info(f"  Keys only in benchmark: {len(bench_keys - our_keys)}")

    if len(merged) == 0:
        logger.error("No common key-weeks found! Check key format alignment.")
        return

    # =========================================================================
    # STEP 5: ACCURACY & BIAS — KEY × WEEK LEVEL
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("[STEP 5] ACCURACY & BIAS COMPARISON — KEY × WEEK LEVEL")
    logger.info("=" * 80)

    actual = merged[TARGET_COL].values
    our_pred = merged['our_forecast'].values
    bench_pred = merged['benchmark_forecast'].values

    our_wape = compute_wape(actual, our_pred)
    bench_wape = compute_wape(actual, bench_pred)
    our_bias = compute_bias(actual, our_pred)
    bench_bias = compute_bias(actual, bench_pred)

    logger.info(f"\n  {'Metric':<25} {'Our Forecast':>15} {'Benchmark (TVF)':>15} {'Difference':>15}")
    logger.info(f"  {'-'*70}")
    logger.info(f"  {'WAPE':<25} {our_wape:>14.1%} {bench_wape:>14.1%} {our_wape-bench_wape:>+14.1%}")
    logger.info(f"  {'Bias %':<25} {our_bias:>14.1f}% {bench_bias:>14.1f}% {our_bias-bench_bias:>+14.1f}%")
    logger.info(f"  {'Total Actual':<25} {actual.sum():>15,.0f}")
    logger.info(f"  {'Total Our Forecast':<25} {our_pred.sum():>15,.0f}")
    logger.info(f"  {'Total Benchmark':<25} {bench_pred.sum():>15,.0f}")
    logger.info(f"  {'Common Keys':<25} {merged[KEY_COL].nunique():>15,}")
    logger.info(f"  {'Common Key-Weeks':<25} {len(merged):>15,}")

    # =========================================================================
    # STEP 6: ACCURACY & BIAS — PRODUCT × WEEK LEVEL (AGGREGATED)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("[STEP 6] ACCURACY & BIAS — PRODUCT × WEEK LEVEL (aggregated over customers)")
    logger.info("=" * 80)

    merged['ProductCode'] = merged[KEY_COL].str.split('_').str[0]

    prod_agg = merged.groupby(['ProductCode', DATE_COL]).agg(
        actual=(TARGET_COL, 'sum'),
        our_forecast=('our_forecast', 'sum'),
        benchmark_forecast=('benchmark_forecast', 'sum'),
    ).reset_index()

    prod_actual = prod_agg['actual'].values
    prod_our = prod_agg['our_forecast'].values
    prod_bench = prod_agg['benchmark_forecast'].values

    prod_our_wape = compute_wape(prod_actual, prod_our)
    prod_bench_wape = compute_wape(prod_actual, prod_bench)
    prod_our_bias = compute_bias(prod_actual, prod_our)
    prod_bench_bias = compute_bias(prod_actual, prod_bench)

    logger.info(f"\n  {'Metric':<25} {'Our Forecast':>15} {'Benchmark (TVF)':>15} {'Difference':>15}")
    logger.info(f"  {'-'*70}")
    logger.info(f"  {'WAPE':<25} {prod_our_wape:>14.1%} {prod_bench_wape:>14.1%} {prod_our_wape-prod_bench_wape:>+14.1%}")
    logger.info(f"  {'Bias %':<25} {prod_our_bias:>14.1f}% {prod_bench_bias:>14.1f}% {prod_our_bias-prod_bench_bias:>+14.1f}%")
    logger.info(f"  {'Products':<25} {prod_agg['ProductCode'].nunique():>15,}")
    logger.info(f"  {'Product-Weeks':<25} {len(prod_agg):>15,}")

    # =========================================================================
    # STEP 7: PER-WEEK BREAKDOWN
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("[STEP 7] PER-WEEK ACCURACY BREAKDOWN")
    logger.info("=" * 80)

    logger.info(f"\n  {'Week':<12} {'Our WAPE':>10} {'Bench WAPE':>12} {'Our Bias%':>10} {'Bench Bias%':>12} {'Winner':>8}")
    logger.info(f"  {'-'*66}")

    our_wins = 0
    for week in sorted(merged[DATE_COL].unique()):
        wk = merged[merged[DATE_COL] == week]
        a = wk[TARGET_COL].values
        o = wk['our_forecast'].values
        b = wk['benchmark_forecast'].values
        ow = compute_wape(a, o)
        bw = compute_wape(a, b)
        ob = compute_bias(a, o)
        bb = compute_bias(a, b)
        winner = "OURS" if ow < bw else "BENCH"
        if ow < bw:
            our_wins += 1
        logger.info(f"  {week:<12} {ow:>9.1%} {bw:>11.1%} {ob:>9.1f}% {bb:>11.1f}% {winner:>8}")

    logger.info(f"\n  We win {our_wins}/{len(merged[DATE_COL].unique())} weeks")

    # =========================================================================
    # STEP 8: PER-PRODUCT BREAKDOWN (top 10 by volume)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("[STEP 8] PER-PRODUCT ACCURACY (top products by actual volume)")
    logger.info("=" * 80)

    prod_summary = merged.groupby('ProductCode').agg(
        total_actual=(TARGET_COL, 'sum'),
        total_our=('our_forecast', 'sum'),
        total_bench=('benchmark_forecast', 'sum'),
        n_keys=(KEY_COL, 'nunique'),
        n_weeks=(DATE_COL, 'nunique'),
    ).reset_index()

    prod_summary['our_wape'] = prod_summary.apply(
        lambda r: compute_wape(
            merged[merged['ProductCode'] == r['ProductCode']][TARGET_COL].values,
            merged[merged['ProductCode'] == r['ProductCode']]['our_forecast'].values,
        ), axis=1)
    prod_summary['bench_wape'] = prod_summary.apply(
        lambda r: compute_wape(
            merged[merged['ProductCode'] == r['ProductCode']][TARGET_COL].values,
            merged[merged['ProductCode'] == r['ProductCode']]['benchmark_forecast'].values,
        ), axis=1)
    prod_summary['our_bias'] = prod_summary.apply(
        lambda r: compute_bias(
            merged[merged['ProductCode'] == r['ProductCode']][TARGET_COL].values,
            merged[merged['ProductCode'] == r['ProductCode']]['our_forecast'].values,
        ), axis=1)
    prod_summary['bench_bias'] = prod_summary.apply(
        lambda r: compute_bias(
            merged[merged['ProductCode'] == r['ProductCode']][TARGET_COL].values,
            merged[merged['ProductCode'] == r['ProductCode']]['benchmark_forecast'].values,
        ), axis=1)

    prod_summary = prod_summary.sort_values('total_actual', ascending=False)

    logger.info(f"\n  {'Product':<16} {'Actual':>10} {'Keys':>5} {'Our WAPE':>10} {'Bench WAPE':>12} {'Our Bias%':>10} {'Bench Bias%':>12} {'Win':>5}")
    logger.info(f"  {'-'*85}")

    our_prod_wins = 0
    for _, r in prod_summary.iterrows():
        win = "OURS" if r['our_wape'] < r['bench_wape'] else "BEN"
        if r['our_wape'] < r['bench_wape']:
            our_prod_wins += 1
        logger.info(f"  {r['ProductCode']:<16} {r['total_actual']:>10,.0f} {r['n_keys']:>5} "
                     f"{r['our_wape']:>9.1%} {r['bench_wape']:>11.1%} "
                     f"{r['our_bias']:>9.1f}% {r['bench_bias']:>11.1f}% {win:>5}")

    logger.info(f"\n  We win {our_prod_wins}/{len(prod_summary)} products")

    # =========================================================================
    # STEP 9: INVESTIGATION — WHERE CAN WE IMPROVE?
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("[STEP 9] INVESTIGATION — AREAS FOR IMPROVEMENT")
    logger.info("=" * 80)

    # Where benchmark beats us most
    prod_summary['wape_gap'] = prod_summary['our_wape'] - prod_summary['bench_wape']
    worst_products = prod_summary.nlargest(5, 'wape_gap')

    logger.info("\n  Products where benchmark beats us most (highest WAPE gap):")
    for _, r in worst_products.iterrows():
        logger.info(f"    {r['ProductCode']}: our={r['our_wape']:.1%} vs bench={r['bench_wape']:.1%} "
                     f"(gap={r['wape_gap']:+.1%}), actual={r['total_actual']:,.0f}, "
                     f"our_bias={r['our_bias']:+.1f}%, bench_bias={r['bench_bias']:+.1f}%")

    # Bias direction analysis
    our_overforecast = (merged['our_forecast'] > merged[TARGET_COL]).mean()
    bench_overforecast = (merged['benchmark_forecast'] > merged[TARGET_COL]).mean()
    logger.info(f"\n  Over-forecast rate: Ours={our_overforecast:.1%}, Benchmark={bench_overforecast:.1%}")

    # Zero-demand analysis
    zero_actual = (merged[TARGET_COL] == 0)
    logger.info(f"\n  Zero-demand rows: {zero_actual.sum():,} / {len(merged):,} ({zero_actual.mean():.1%})")
    if zero_actual.any():
        our_on_zeros = merged.loc[zero_actual, 'our_forecast'].mean()
        bench_on_zeros = merged.loc[zero_actual, 'benchmark_forecast'].mean()
        logger.info(f"  Avg forecast on zero-demand rows: Ours={our_on_zeros:.1f}, Benchmark={bench_on_zeros:.1f}")

    nonzero_actual = ~zero_actual
    if nonzero_actual.any():
        our_wape_nz = compute_wape(merged.loc[nonzero_actual, TARGET_COL].values,
                                    merged.loc[nonzero_actual, 'our_forecast'].values)
        bench_wape_nz = compute_wape(merged.loc[nonzero_actual, TARGET_COL].values,
                                      merged.loc[nonzero_actual, 'benchmark_forecast'].values)
        logger.info(f"  WAPE on non-zero demand only: Ours={our_wape_nz:.1%}, Benchmark={bench_wape_nz:.1%}")

    # Volume bucket analysis
    try:
        merged['volume_bucket'] = pd.cut(merged[TARGET_COL],
                                          bins=[-0.01, 0, 50, 500, float('inf')],
                                          labels=['Zero', 'Low (1-50)', 'Medium (51-500)', 'High (500+)'])
        logger.info(f"\n  WAPE by volume bucket:")
        logger.info(f"  {'Bucket':<20} {'N':>6} {'Our WAPE':>10} {'Bench WAPE':>12} {'Gap':>8}")
        logger.info(f"  {'-'*58}")
        for bucket in merged['volume_bucket'].cat.categories:
            bk = merged[merged['volume_bucket'] == bucket]
            if len(bk) < 5:
                continue
            ow = compute_wape(bk[TARGET_COL].values, bk['our_forecast'].values)
            bw = compute_wape(bk[TARGET_COL].values, bk['benchmark_forecast'].values)
            logger.info(f"  {str(bucket):<20} {len(bk):>6} {ow:>9.1%} {bw:>11.1%} {ow-bw:>+7.1%}")
    except Exception as e:
        logger.warning(f"  Volume bucket analysis skipped: {e}")

    # =========================================================================
    # STEP 10: RECOMMENDATIONS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("[STEP 10] RECOMMENDATIONS FOR IMPROVEMENT")
    logger.info("=" * 80)

    recommendations = []

    if our_bias > 10:
        recommendations.append("HIGH POSITIVE BIAS: Our forecasts systematically over-predict. "
                                "Apply negative bias calibration or use quantile regression at q=0.45.")
    elif our_bias < -10:
        recommendations.append("HIGH NEGATIVE BIAS: Our forecasts systematically under-predict. "
                                "Apply positive bias calibration or use quantile regression at q=0.55.")

    if our_wape > bench_wape * 1.1:
        recommendations.append(f"WAPE GAP: Our WAPE ({our_wape:.1%}) is worse than benchmark ({bench_wape:.1%}). "
                                "Consider: more features (hierarchy, external), ensemble methods, "
                                "or longer training history.")

    zero_pct = zero_actual.mean()
    if zero_pct > 0.5:
        recommendations.append(f"HIGH SPARSITY: {zero_pct:.0%} of demand is zero. "
                                "Use zero-inflated/hurdle models or quantile median regression. "
                                "The benchmark likely uses domain-specific zero handling.")

    if our_overforecast > 0.6:
        recommendations.append("OVER-FORECAST TENDENCY: We predict demand when there is none. "
                                "Add intermittency features (weeks_since_demand, demand_probability). "
                                "Use Tweedie loss or asymmetric loss penalizing false positives more.")

    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")

    if not recommendations:
        logger.info("  No specific issues found — forecast quality is competitive with benchmark.")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - t0
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Category:                DACH SKIN CARE")
    logger.info(f"  Backtest period:         {TARGET_WEEKS_START} to {TARGET_WEEKS_END} (lag 4)")
    logger.info(f"  Common keys:             {merged[KEY_COL].nunique()}")
    logger.info(f"  Common products:         {merged['ProductCode'].nunique()}")
    logger.info(f"")
    logger.info(f"  KEY × WEEK LEVEL:")
    logger.info(f"    Our WAPE:              {our_wape:.1%}")
    logger.info(f"    Benchmark WAPE:        {bench_wape:.1%}")
    logger.info(f"    Our Bias:              {our_bias:+.1f}%")
    logger.info(f"    Benchmark Bias:        {bench_bias:+.1f}%")
    logger.info(f"")
    logger.info(f"  PRODUCT × WEEK LEVEL:")
    logger.info(f"    Our WAPE:              {prod_our_wape:.1%}")
    logger.info(f"    Benchmark WAPE:        {prod_bench_wape:.1%}")
    logger.info(f"    Our Bias:              {prod_our_bias:+.1f}%")
    logger.info(f"    Benchmark Bias:        {prod_bench_bias:+.1f}%")
    logger.info(f"")
    logger.info(f"  Execution time:          {total_time:.1f}s")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
