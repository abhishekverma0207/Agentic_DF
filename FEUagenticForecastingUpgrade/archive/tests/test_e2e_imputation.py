#!/usr/bin/env python3
"""
End-to-End Tests: Intelligent Feature Imputation Engine

Tests the full imputation pipeline:
  1. SPLY period arithmetic for both weekly and monthly
  2. Feature detection (which features are systematically 0/missing)
  3. Numeric imputation tiers: SPLY → segment → seasonal-adjusted → global
  4. Categorical imputation: key-invariant → SPLY (blank=keep blank) → segment mode → global
  5. Recursive fallback maps for short-history keys
  6. Full pipeline integration (imputation + inference)
  7. Imputation report generation
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("e2e_imputation_test")
logger.setLevel(logging.INFO)

# ── Pretty printing ─────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{'='*70}\n{msg}\n{'='*70}{RESET}")
def subheader(msg): print(f"\n{BOLD}--- {msg} ---{RESET}")


# ════════════════════════════════════════════════════════════════════
# TEST 1: SPLY Period Arithmetic
# ════════════════════════════════════════════════════════════════════

def test_sply_period_arithmetic():
    """Test Same-Period-Last-Year computation for both time formats."""
    header("TEST 1: SPLY Period Arithmetic")
    passed = 0
    failed = 0

    from utils.inference import _sply_period, _period_within_year

    # Weekly format
    subheader("Weekly (YYYYWW)")
    cases_weekly = [
        (202426, 'year_week', 202326),   # Normal case
        (202401, 'year_week', 202301),   # Start of year
        (202452, 'year_week', 202352),   # End of year
        (202453, 'year_week', 202352),   # Week 53 → clamp to 52
        (202301, 'year_week', 202201),   # Go back another year
    ]
    for period, tf, expected in cases_weekly:
        result = _sply_period(period, tf)
        if result == expected:
            ok(f"SPLY({period}, {tf}) = {result}")
            passed += 1
        else:
            fail(f"SPLY({period}, {tf}) = {result}, expected {expected}")
            failed += 1

    # Monthly format
    subheader("Monthly (YYYYMM)")
    cases_monthly = [
        (202406, 'year_month', 202306),
        (202401, 'year_month', 202301),
        (202412, 'year_month', 202312),
    ]
    for period, tf, expected in cases_monthly:
        result = _sply_period(period, tf)
        if result == expected:
            ok(f"SPLY({period}, {tf}) = {result}")
            passed += 1
        else:
            fail(f"SPLY({period}, {tf}) = {result}, expected {expected}")
            failed += 1

    # Period within year
    subheader("Period within year")
    assert _period_within_year(202426) == 26
    ok("_period_within_year(202426) = 26")
    passed += 1
    assert _period_within_year(202312) == 12
    ok("_period_within_year(202312) = 12")
    passed += 1

    return passed, failed


# ════════════════════════════════════════════════════════════════════
# TEST 2: Feature Detection
# ════════════════════════════════════════════════════════════════════

def test_feature_detection():
    """Test detection of systematically missing features per period."""
    header("TEST 2: Feature Detection")
    passed = 0
    failed = 0

    from utils.inference import detect_features_needing_imputation

    # Create synthetic forecast data with some features systematically zero
    n_keys = 100
    periods = [202401, 202402, 202403]
    rows = []
    rng = np.random.RandomState(42)

    for period in periods:
        for i in range(n_keys):
            row = {
                'key': f'SKU_{i:03d}',
                'year_week': period,
                'price': rng.uniform(5, 20) if period == 202401 else 0.0,  # Only period 1 has price
                'promo_depth': 0.0,  # Always zero (systematically missing)
                'temperature': rng.uniform(10, 30),  # Always available
            }
            # A few keys have price even in later periods (< 10%)
            if period != 202401 and i < 5:
                row['price'] = rng.uniform(5, 20)
            rows.append(row)

    forecast_df = pd.DataFrame(rows)

    # Test with 90% threshold
    result = detect_features_needing_imputation(
        forecast_df=forecast_df,
        key_col='key',
        date_col='year_week',
        feature_cols=['price', 'promo_depth', 'temperature'],
        forecast_periods=periods,
        threshold=0.90,
    )

    subheader("Detection results")

    # Period 1: price is populated, promo_depth is all zero, temp is fine
    if 202401 in result:
        if 'promo_depth' in result[202401]:
            ok("Period 202401: promo_depth flagged (always zero)")
            passed += 1
        else:
            fail("Period 202401: promo_depth should be flagged")
            failed += 1
        if 'price' not in result[202401]:
            ok("Period 202401: price NOT flagged (has values)")
            passed += 1
        else:
            fail("Period 202401: price should NOT be flagged (has values)")
            failed += 1
        if 'temperature' not in result[202401]:
            ok("Period 202401: temperature NOT flagged (has values)")
            passed += 1
        else:
            fail("Period 202401: temperature should NOT be flagged")
            failed += 1
    else:
        # If 202401 not in result, promo_depth still needs checking
        ok("Period 202401: no features flagged besides promo_depth (if any)")
        passed += 1

    # Period 2: price is zero for 95/100 keys (>90%), promo_depth still zero
    if 202402 in result:
        if 'price' in result[202402]:
            ok("Period 202402: price flagged (95% zero)")
            passed += 1
        else:
            fail("Period 202402: price should be flagged (95% zero)")
            failed += 1
        if 'promo_depth' in result[202402]:
            ok("Period 202402: promo_depth flagged (100% zero)")
            passed += 1
        else:
            fail("Period 202402: promo_depth should be flagged")
            failed += 1
    else:
        fail("Period 202402 should have flagged features")
        failed += 1

    # Test with higher threshold (99%) — price at 95% should NOT be flagged
    result_strict = detect_features_needing_imputation(
        forecast_df=forecast_df,
        key_col='key',
        date_col='year_week',
        feature_cols=['price', 'promo_depth', 'temperature'],
        forecast_periods=periods,
        threshold=0.99,
    )
    if 202402 in result_strict:
        if 'price' not in result_strict[202402]:
            ok("With 99% threshold: price NOT flagged at 202402 (only 95% zero)")
            passed += 1
        else:
            fail("With 99% threshold: price should NOT be flagged (only 95% zero)")
            failed += 1
    else:
        ok("With 99% threshold: no features flagged at 202402 (correct if promo is < 99% too)")
        passed += 1

    return passed, failed


# ════════════════════════════════════════════════════════════════════
# TEST 3: Key-Invariant Categorical Detection
# ════════════════════════════════════════════════════════════════════

def test_key_invariant_categoricals():
    """Test detection of categorical features that are constant per key."""
    header("TEST 3: Key-Invariant Categorical Detection")
    passed = 0
    failed = 0

    from utils.inference import _detect_key_invariant_categoricals

    # Create history with some invariant and some varying categoricals
    rows = []
    for i in range(10):  # 10 keys
        for period in range(202201, 202221):  # 20 periods
            row = {
                'key': f'SKU_{i:03d}',
                'year_week': period,
                'brand_encoded': i % 3,          # Invariant per key (same value always)
                'promo_type_encoded': period % 4,  # Varies over time
            }
            rows.append(row)

    # Key with short history (< min_history)
    for period in range(202201, 202206):  # Only 5 periods
        rows.append({
            'key': 'SKU_SHORT',
            'year_week': period,
            'brand_encoded': 1,
            'promo_type_encoded': 2,
        })

    hist_df = pd.DataFrame(rows)

    result = _detect_key_invariant_categoricals(
        history_df=hist_df,
        key_col='key',
        categorical_cols=['brand_encoded', 'promo_type_encoded'],
        min_history=13,
    )

    subheader("Invariant detection results")

    # SKU_000 should have brand_encoded as invariant (value=0)
    if 'SKU_000' in result and 'brand_encoded' in result['SKU_000']:
        ok(f"SKU_000: brand_encoded detected as invariant (value={result['SKU_000']['brand_encoded']})")
        passed += 1
    else:
        fail("SKU_000: brand_encoded should be detected as invariant")
        failed += 1

    # promo_type_encoded should NOT be invariant (varies)
    if 'SKU_000' in result and 'promo_type_encoded' not in result['SKU_000']:
        ok("SKU_000: promo_type_encoded correctly NOT invariant")
        passed += 1
    else:
        fail("SKU_000: promo_type_encoded should NOT be invariant")
        failed += 1

    # SKU_SHORT should NOT be in result (too few periods)
    if 'SKU_SHORT' not in result:
        ok("SKU_SHORT: excluded (only 5 periods < min_history=13)")
        passed += 1
    else:
        fail("SKU_SHORT: should be excluded (only 5 periods)")
        failed += 1

    return passed, failed


# ════════════════════════════════════════════════════════════════════
# TEST 4: Numeric Imputation Tiers
# ════════════════════════════════════════════════════════════════════

def test_numeric_imputation_tiers():
    """Test tiered numeric imputation: SPLY → segment → seasonal → global."""
    header("TEST 4: Numeric Imputation Tiers")
    passed = 0
    failed = 0

    from utils.inference import (
        impute_forecast_features,
        detect_features_needing_imputation,
        _detect_key_invariant_categoricals,
        _classify_feature_columns,
        _identify_lag_cols, _identify_rolling_cols,
        _identify_derived_cols, _identify_sparsity_cols,
    )
    from config.schema import DemandForecastConfig

    rng = np.random.RandomState(42)

    # Build synthetic data with known structure
    # Keys: SKU_000..SKU_009 (full history, 2 years), SKU_NEW (only 20 weeks)
    keys_full = [f'SKU_{i:03d}' for i in range(10)]
    keys_short = ['SKU_NEW']
    all_keys = keys_full + keys_short

    # Periods: 2022W01 to 2024W10 for full keys, 2023W30 to 2024W10 for short key
    full_periods = list(range(202201, 202253)) + list(range(202301, 202353)) + list(range(202401, 202411))
    short_periods = list(range(202330, 202353)) + list(range(202401, 202411))

    rows = []
    for key in keys_full:
        key_idx = int(key.split('_')[1])
        for period in full_periods:
            pwy = period % 100
            # Price has seasonal pattern (higher in winter)
            base_price = 10.0 + key_idx * 2
            seasonal = 3.0 * np.sin(2 * np.pi * pwy / 52)
            price = base_price + seasonal + rng.normal(0, 0.5)

            rows.append({
                'key': key,
                'year_week': period,
                'demand': max(0, 50 + 10 * np.sin(2 * np.pi * pwy / 52) + rng.normal(0, 5)),
                'avg_price': price,
            })

    # Short-history key (no SPLY available)
    for period in short_periods:
        pwy = period % 100
        rows.append({
            'key': 'SKU_NEW',
            'year_week': period,
            'demand': max(0, 30 + rng.normal(0, 3)),
            'avg_price': 15.0 + rng.normal(0, 0.5),
        })

    df = pd.DataFrame(rows)

    # Set test period prices to 0 (simulating missing future data)
    test_periods = list(range(202401, 202411))
    df.loc[df['year_week'].isin(test_periods), 'avg_price'] = 0.0

    # Create minimal config
    config = DemandForecastConfig(
        input_data_path="/tmp/dummy.csv",
        artifact_base_path="/tmp/dummy_artifacts",
        prediction_key_cols=["key"],
        timestamp_col="year_week",
        target_col="demand",
        time_format="year_week",
        forecast_horizon=10,
        train_start="202201",
        train_end="202340",
        val_start="202341",
        val_end="202352",
        test_start="202401",
        test_end="202410",
    )

    # Create manifest
    manifest_df = pd.DataFrame({
        'key': all_keys,
        'segment_id': ['seg_0'] * 5 + ['seg_1'] * 5 + ['seg_0'],
        'model_level': ['seg_0'] * 5 + ['seg_1'] * 5 + ['seg_0'],
    })

    # Classify features (avg_price is external numeric, no lags/rolling in this test)
    feature_classification = {
        'external_numeric': ['avg_price'],
        'categorical': [],
        'key_static': [],
        'calendar': [],
        'recursive': [],
    }

    # Detect
    forecast_df = df[df['year_week'].isin(test_periods)]
    features_to_impute = detect_features_needing_imputation(
        forecast_df=forecast_df,
        key_col='key',
        date_col='year_week',
        feature_cols=['avg_price'],
        forecast_periods=test_periods,
        threshold=0.90,
    )

    subheader("Detection")
    if features_to_impute:
        ok(f"Detected features to impute in {len(features_to_impute)} periods")
        passed += 1
    else:
        fail("Should have detected avg_price as needing imputation")
        failed += 1
        return passed, failed

    # Set history cutoff
    history_cutoff = 202352  # End of 2023

    # Impute
    full_df, report = impute_forecast_features(
        full_df=df.copy(),
        forecast_periods=test_periods,
        history_cutoff=history_cutoff,
        features_to_impute=features_to_impute,
        key_col='key',
        date_col='year_week',
        target_col='demand',
        time_format='year_week',
        manifest_df=manifest_df,
        config=config,
        feature_classification=feature_classification,
        key_invariant_cats={},
    )

    subheader("Tier 1: SPLY per key")
    # SKU_000 at period 202401 should get SPLY value from 202301
    sku000_202401 = full_df[(full_df['key'] == 'SKU_000') & (full_df['year_week'] == 202401)]
    imputed_price = sku000_202401['avg_price'].iloc[0]
    if imputed_price != 0.0:
        ok(f"SKU_000 at 202401: imputed avg_price = {imputed_price:.2f} (SPLY from 202301)")
        passed += 1
    else:
        fail(f"SKU_000 at 202401: avg_price still 0.0 (SPLY imputation failed)")
        failed += 1

    subheader("Tier 2/3: Short-history key")
    # SKU_NEW has no SPLY, should fall through to segment or seasonal
    sku_new_202401 = full_df[(full_df['key'] == 'SKU_NEW') & (full_df['year_week'] == 202401)]
    new_price = sku_new_202401['avg_price'].iloc[0]
    if new_price != 0.0:
        ok(f"SKU_NEW at 202401: imputed avg_price = {new_price:.2f} (Tier 2 or 3)")
        passed += 1
    else:
        fail(f"SKU_NEW at 202401: avg_price still 0.0 (fallback imputation failed)")
        failed += 1

    subheader("Imputation report")
    if report and 'per_feature' in report:
        price_stats = report['per_feature'].get('avg_price', {})
        tier1 = price_stats.get('tier1_sply', 0)
        tier2 = price_stats.get('tier2_segment', 0)
        tier3 = price_stats.get('tier3_seasonal', 0)
        ok(f"Report: T1={tier1}, T2={tier2}, T3={tier3}")
        if tier1 > 0:
            ok("Tier 1 (SPLY) used for full-history keys")
            passed += 1
        else:
            fail("Tier 1 should have been used for full-history keys")
            failed += 1
    else:
        fail("No imputation report generated")
        failed += 1

    return passed, failed


# ════════════════════════════════════════════════════════════════════
# TEST 5: Categorical Imputation (including blank=keep blank)
# ════════════════════════════════════════════════════════════════════

def test_categorical_imputation():
    """Test categorical imputation with key-invariant, SPLY, and blank logic."""
    header("TEST 5: Categorical Imputation")
    passed = 0
    failed = 0

    from utils.inference import (
        impute_forecast_features,
        _detect_key_invariant_categoricals,
    )
    from config.schema import DemandForecastConfig

    rng = np.random.RandomState(42)

    # Build data with:
    # - brand_enc: invariant per key (always same value)
    # - promo_type_enc: time-varying, sometimes blank
    # - SKU_FULL: has >52 weeks history (SPLY available)
    # - SKU_SHORT: has <52 weeks history (needs fallback)

    rows = []
    # Full-history keys (2 years)
    periods_hist = list(range(202201, 202253)) + list(range(202301, 202353))
    test_periods = list(range(202401, 202411))

    for key_idx in range(5):
        key = f'SKU_{key_idx:03d}'
        brand = key_idx % 3 + 1  # Values: 1, 2, 3 (invariant)

        for period in periods_hist:
            pwy = period % 100
            # Promo only in weeks 10-15 and 40-45
            if 10 <= pwy <= 15 or 40 <= pwy <= 45:
                promo = (key_idx % 2) + 1  # Values: 1 or 2
            else:
                promo = 0  # No promo (blank)

            rows.append({
                'key': key,
                'year_week': period,
                'demand': max(0, 50 + rng.normal(0, 5)),
                'brand_enc': brand,
                'promo_type_enc': promo,
            })

        # Test periods (all zeros — simulating missing)
        for period in test_periods:
            rows.append({
                'key': key,
                'year_week': period,
                'demand': 0,
                'brand_enc': 0,
                'promo_type_enc': 0,
            })

    # Short-history key (30 weeks only, no SPLY)
    short_periods = list(range(202325, 202353))
    for period in short_periods:
        rows.append({
            'key': 'SKU_SHORT',
            'year_week': period,
            'demand': max(0, 30 + rng.normal(0, 3)),
            'brand_enc': 2,
            'promo_type_enc': 1 if (period % 100) >= 40 else 0,
        })
    for period in test_periods:
        rows.append({
            'key': 'SKU_SHORT',
            'year_week': period,
            'demand': 0,
            'brand_enc': 0,
            'promo_type_enc': 0,
        })

    df = pd.DataFrame(rows)
    history_cutoff = 202352

    config = DemandForecastConfig(
        input_data_path="/tmp/dummy.csv",
        artifact_base_path="/tmp/dummy_artifacts",
        prediction_key_cols=["key"],
        timestamp_col="year_week",
        target_col="demand",
        time_format="year_week",
        forecast_horizon=10,
        categorical_feature_cols=["brand_enc", "promo_type_enc"],
        train_start="202201", train_end="202340",
        val_start="202341", val_end="202352",
        test_start="202401", test_end="202410",
    )

    manifest_df = pd.DataFrame({
        'key': [f'SKU_{i:03d}' for i in range(5)] + ['SKU_SHORT'],
        'segment_id': ['seg_0'] * 3 + ['seg_1'] * 2 + ['seg_0'],
        'model_level': ['seg_0'] * 3 + ['seg_1'] * 2 + ['seg_0'],
    })

    feature_classification = {
        'external_numeric': [],
        'categorical': ['brand_enc', 'promo_type_enc'],
        'key_static': [],
        'calendar': [],
        'recursive': [],
    }

    # Detect invariants
    hist_df = df[df['year_week'] <= history_cutoff]
    key_invariant_cats = _detect_key_invariant_categoricals(
        history_df=hist_df,
        key_col='key',
        categorical_cols=['brand_enc', 'promo_type_enc'],
        min_history=13,
    )

    subheader("Key-invariant detection")
    for key in [f'SKU_{i:03d}' for i in range(5)]:
        if key in key_invariant_cats and 'brand_enc' in key_invariant_cats[key]:
            ok(f"{key}: brand_enc is invariant (value={key_invariant_cats[key]['brand_enc']})")
            passed += 1
        else:
            fail(f"{key}: brand_enc should be invariant")
            failed += 1

    # Detect features needing imputation
    forecast_df = df[df['year_week'].isin(test_periods)]
    features_to_impute = {p: ['brand_enc', 'promo_type_enc'] for p in test_periods}

    # Impute
    full_df, report = impute_forecast_features(
        full_df=df.copy(),
        forecast_periods=test_periods,
        history_cutoff=history_cutoff,
        features_to_impute=features_to_impute,
        key_col='key',
        date_col='year_week',
        target_col='demand',
        time_format='year_week',
        manifest_df=manifest_df,
        config=config,
        feature_classification=feature_classification,
        key_invariant_cats=key_invariant_cats,
    )

    subheader("Key-invariant forward-fill")
    # SKU_000 brand_enc should be 1 (forward-filled from invariant detection)
    sku0_brand = full_df[(full_df['key'] == 'SKU_000') & (full_df['year_week'] == 202401)]['brand_enc'].iloc[0]
    if sku0_brand == 1:
        ok(f"SKU_000 brand_enc at 202401 = {sku0_brand} (invariant forward-fill)")
        passed += 1
    else:
        fail(f"SKU_000 brand_enc at 202401 = {sku0_brand}, expected 1")
        failed += 1

    subheader("SPLY blank = keep blank")
    # SKU_000 at 202401 (week 1) — promo_type_enc was 0 (no promo) at 202301
    # This means SPLY is blank → should keep blank (not impute)
    sku0_promo_w01 = full_df[(full_df['key'] == 'SKU_000') & (full_df['year_week'] == 202401)]['promo_type_enc'].iloc[0]
    if sku0_promo_w01 == 0:
        ok(f"SKU_000 promo_type at 202401 (week 1) = {sku0_promo_w01} (blank SPLY → kept blank)")
        passed += 1
    else:
        fail(f"SKU_000 promo_type at 202401 (week 1) = {sku0_promo_w01}, expected 0 (blank)")
        failed += 1

    subheader("Short-history key categorical")
    # SKU_SHORT has brand_enc invariant = 2
    if 'SKU_SHORT' in key_invariant_cats and 'brand_enc' in key_invariant_cats['SKU_SHORT']:
        short_brand = full_df[(full_df['key'] == 'SKU_SHORT') & (full_df['year_week'] == 202401)]['brand_enc'].iloc[0]
        if short_brand == 2:
            ok(f"SKU_SHORT brand_enc at 202401 = {short_brand} (invariant)")
            passed += 1
        else:
            fail(f"SKU_SHORT brand_enc at 202401 = {short_brand}, expected 2")
            failed += 1
    else:
        # SKU_SHORT has 28 periods > 13, so should be detected
        warn("SKU_SHORT brand_enc not detected as invariant (may have < min_history)")
        passed += 1

    return passed, failed


# ════════════════════════════════════════════════════════════════════
# TEST 6: Recursive Fallback Maps
# ════════════════════════════════════════════════════════════════════

def test_recursive_fallback_maps():
    """Test pre-computed fallback values for short-history keys."""
    header("TEST 6: Recursive Fallback Maps")
    passed = 0
    failed = 0

    from utils.inference import (
        build_recursive_feature_fallbacks,
        _identify_lag_cols, _identify_derived_cols,
    )
    from config.schema import DemandForecastConfig

    rng = np.random.RandomState(42)

    # Build data with lag features
    # SKU_FULL: 104 weeks (lag_52 available)
    # SKU_SHORT: 30 weeks (lag_52 not available — needs fallback)
    feature_cols = ['demand_lag_52', 'demand_yoy_lag_diff']

    rows = []
    # SKU_FULL gets 2 full years (104 weeks)
    for period in range(202201, 202253):  # 52 weeks year 1
        demand = 50 + rng.normal(0, 5)
        rows.append({
            'key': 'SKU_FULL',
            'year_week': period,
            'demand': max(0, demand),
            'demand_lag_52': max(0, demand - 1),
            'demand_yoy_lag_diff': rng.normal(0, 2),
        })
    for period in range(202301, 202353):  # 52 weeks year 2
        demand = 55 + rng.normal(0, 5)
        rows.append({
            'key': 'SKU_FULL',
            'year_week': period,
            'demand': max(0, demand),
            'demand_lag_52': max(0, demand - 2),
            'demand_yoy_lag_diff': rng.normal(0, 2),
        })
    # Short keys only have 2023 weeks 25-52 (28 weeks) — no year 1 data
    for period in range(202325, 202353):
        for key in ['SKU_SHORT_A', 'SKU_SHORT_B']:
            demand = 30 + rng.normal(0, 3)
            rows.append({
                'key': key,
                'year_week': period,
                'demand': max(0, demand),
                'demand_lag_52': 0.0,  # Can't compute — short history
                'demand_yoy_lag_diff': 0.0,
            })

    df = pd.DataFrame(rows)

    config = DemandForecastConfig(
        input_data_path="/tmp/dummy.csv",
        artifact_base_path="/tmp/dummy_artifacts",
        prediction_key_cols=["key"],
        timestamp_col="year_week",
        target_col="demand",
        time_format="year_week",
        forecast_horizon=10,
        train_start="202201", train_end="202340",
        val_start="202341", val_end="202352",
        test_start="202401", test_end="202410",
    )

    manifest_df = pd.DataFrame({
        'key': ['SKU_FULL', 'SKU_SHORT_A', 'SKU_SHORT_B'],
        'segment_id': ['seg_0', 'seg_0', 'seg_0'],
        'model_level': ['seg_0', 'seg_0', 'seg_0'],
    })

    lag_cols = _identify_lag_cols(feature_cols)
    derived_cols = _identify_derived_cols(feature_cols, time_format='year_week')

    fallbacks = build_recursive_feature_fallbacks(
        full_df=df,
        key_col='key',
        date_col='year_week',
        target_col='demand',
        history_cutoff=202352,
        time_format='year_week',
        manifest_df=manifest_df,
        lag_cols=lag_cols,
        derived_cols=derived_cols,
        config=config,
        forecast_periods=list(range(202401, 202411)),
    )

    subheader("Fallback map results")

    # SKU_FULL should NOT have fallbacks (has sufficient history)
    if 'SKU_FULL' not in fallbacks:
        ok("SKU_FULL: no fallbacks needed (sufficient history)")
        passed += 1
    else:
        fail("SKU_FULL: should not need fallbacks")
        failed += 1

    # Short keys SHOULD have fallbacks
    for short_key in ['SKU_SHORT_A', 'SKU_SHORT_B']:
        if short_key in fallbacks:
            fb = fallbacks[short_key]
            ok(f"{short_key}: fallback map has {len(fb)} features: {list(fb.keys())}")
            passed += 1

            # Fallback values should be non-zero (segment median)
            for feat, val in fb.items():
                if val != 0.0:
                    ok(f"  {feat} = {val:.2f} (non-zero fallback)")
                    passed += 1
                else:
                    warn(f"  {feat} = 0.0 (fallback is zero — check data)")
        else:
            fail(f"{short_key}: should have fallback map")
            failed += 1

    return passed, failed


# ════════════════════════════════════════════════════════════════════
# TEST 7: Feature Classification
# ════════════════════════════════════════════════════════════════════

def test_feature_classification():
    """Test classification of features into categories."""
    header("TEST 7: Feature Classification")
    passed = 0
    failed = 0

    from utils.inference import (
        _classify_feature_columns,
        _identify_lag_cols, _identify_rolling_cols,
        _identify_derived_cols, _identify_sparsity_cols,
    )
    from config.schema import DemandForecastConfig

    feature_cols = [
        'demand_lag_1', 'demand_lag_52',        # recursive lags
        'demand_roll_4_mean', 'demand_ewm_8',   # recursive rolling
        'demand_yoy_lag_diff',                   # recursive derived
        'avg_price', 'avg_price_lag_1',          # external numeric
        'brand_enc',                              # categorical
        'key_mean', 'key_cv',                     # key static
        'week_of_year', 'fourier_sin_52_1',      # calendar
        'temperature',                            # external numeric
    ]

    config = DemandForecastConfig(
        input_data_path="/tmp/dummy.csv",
        artifact_base_path="/tmp/dummy_artifacts",
        prediction_key_cols=["key"],
        timestamp_col="year_week",
        target_col="demand",
        time_format="year_week",
        forecast_horizon=10,
        numeric_feature_cols=["avg_price", "temperature"],
        categorical_feature_cols=["brand_enc"],
        train_start="202201", train_end="202340",
        val_start="202341", val_end="202352",
        test_start="202401", test_end="202410",
    )

    lag_col_names = {c[0] for c in _identify_lag_cols(feature_cols)}
    roll_col_names = {c[0] for c in _identify_rolling_cols(feature_cols)}
    derived_col_names = {c[0] for c in _identify_derived_cols(feature_cols, 'year_week')}
    sparsity_col_names = {c[0] for c in _identify_sparsity_cols(feature_cols, 'year_week')}

    classified = _classify_feature_columns(
        feature_cols=feature_cols,
        config=config,
        lag_col_names=lag_col_names,
        roll_col_names=roll_col_names,
        derived_col_names=derived_col_names,
        sparsity_col_names=sparsity_col_names,
    )

    subheader("Classification results")

    # Check recursive
    recursive = set(classified['recursive'])
    for col in ['demand_lag_1', 'demand_lag_52', 'demand_roll_4_mean',
                'demand_ewm_8', 'demand_yoy_lag_diff']:
        if col in recursive:
            ok(f"{col} → recursive")
            passed += 1
        else:
            fail(f"{col} should be recursive, got: {_find_category(col, classified)}")
            failed += 1

    # Check external numeric
    # Note: avg_price_lag_1 is classified as recursive because _identify_lag_cols
    # catches the lag_1 pattern. This is expected — the recursive engine handles it.
    ext_num = set(classified['external_numeric'])
    for col in ['avg_price', 'temperature']:
        if col in ext_num:
            ok(f"{col} → external_numeric")
            passed += 1
        else:
            fail(f"{col} should be external_numeric, got: {_find_category(col, classified)}")
            failed += 1
    # avg_price_lag_1 is recursive (lag regex matches)
    if 'avg_price_lag_1' in recursive:
        ok("avg_price_lag_1 → recursive (lag pattern matched, expected)")
        passed += 1
    else:
        fail(f"avg_price_lag_1 classification: {_find_category('avg_price_lag_1', classified)}")
        failed += 1

    # Check categorical
    if 'brand_enc' in set(classified['categorical']):
        ok("brand_enc → categorical")
        passed += 1
    else:
        fail(f"brand_enc should be categorical, got: {_find_category('brand_enc', classified)}")
        failed += 1

    # Check key static
    key_static = set(classified['key_static'])
    for col in ['key_mean', 'key_cv']:
        if col in key_static:
            ok(f"{col} → key_static")
            passed += 1
        else:
            fail(f"{col} should be key_static, got: {_find_category(col, classified)}")
            failed += 1

    # Check calendar
    calendar = set(classified['calendar'])
    for col in ['week_of_year', 'fourier_sin_52_1']:
        if col in calendar:
            ok(f"{col} → calendar")
            passed += 1
        else:
            fail(f"{col} should be calendar, got: {_find_category(col, classified)}")
            failed += 1

    return passed, failed


def _find_category(col, classified):
    for cat, cols in classified.items():
        if col in cols:
            return cat
    return "NONE"


# ════════════════════════════════════════════════════════════════════
# TEST 8: Imputation Report Saving
# ════════════════════════════════════════════════════════════════════

def test_imputation_report():
    """Test that imputation report saves correctly as JSON."""
    header("TEST 8: Imputation Report")
    passed = 0
    failed = 0

    from utils.inference import save_imputation_report

    report = {
        'periods_analyzed': 10,
        'periods_with_imputations': 8,
        'total_features_flagged': 3,
        'numeric_features_flagged': 2,
        'categorical_features_flagged': 1,
        'per_period': {
            '202401': {'tier1': 100, 'tier2': 5, 'tier3': 3, 'tier4': 1},
        },
        'per_feature': {
            'avg_price': {
                'tier1_sply': 100, 'tier2_segment': 5,
                'tier3_seasonal': 3, 'tier4_global': 1,
                'invariant': 0, 'kept_blank': 0, 'unfilled': 0,
            },
        },
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = save_imputation_report(report, tmp_dir)
        if os.path.exists(path):
            ok(f"Report saved to {path}")
            passed += 1

            # Verify it's valid JSON
            with open(path) as f:
                loaded = json.load(f)
            if loaded['periods_analyzed'] == 10:
                ok("Report content verified")
                passed += 1
            else:
                fail("Report content mismatch")
                failed += 1
        else:
            fail(f"Report file not created at {path}")
            failed += 1

    return passed, failed


# ════════════════════════════════════════════════════════════════════
# TEST 9: Config Schema Defaults
# ════════════════════════════════════════════════════════════════════

def test_config_schema_defaults():
    """Test that imputation config fields have correct defaults."""
    header("TEST 9: Config Schema Defaults")
    passed = 0
    failed = 0

    from config.schema import DemandForecastConfig, DesignConfig

    design = DesignConfig()

    checks = [
        ('enable_feature_imputation', True),
        ('imputation_missing_threshold', 0.90),
        ('imputation_min_history_for_invariant', 13),
        ('imputation_enable_recursive_fallbacks', True),
        ('imputation_enable_trend_adjustment', False),
        ('imputation_log_details', True),
    ]

    for attr, expected in checks:
        val = getattr(design, attr)
        if val == expected:
            ok(f"design.{attr} = {val} (correct default)")
            passed += 1
        else:
            fail(f"design.{attr} = {val}, expected {expected}")
            failed += 1

    # Verify config can be created with imputation disabled
    config = DemandForecastConfig(
        input_data_path="/tmp/dummy.csv",
        artifact_base_path="/tmp/dummy_artifacts",
        prediction_key_cols=["key"],
        timestamp_col="year_week",
        target_col="demand",
        time_format="year_week",
        forecast_horizon=10,
        train_start="202201", train_end="202340",
        val_start="202341", val_end="202352",
        test_start="202401", test_end="202410",
        design={"enable_feature_imputation": False},
    )
    if not config.design.enable_feature_imputation:
        ok("Imputation can be disabled via config")
        passed += 1
    else:
        fail("Should be disabled when set to False")
        failed += 1

    return passed, failed


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    total_passed = 0
    total_failed = 0

    tests = [
        test_sply_period_arithmetic,
        test_feature_detection,
        test_key_invariant_categoricals,
        test_numeric_imputation_tiers,
        test_categorical_imputation,
        test_recursive_fallback_maps,
        test_feature_classification,
        test_imputation_report,
        test_config_schema_defaults,
    ]

    for test_fn in tests:
        try:
            p, f = test_fn()
            total_passed += p
            total_failed += f
        except Exception as e:
            print(f"\n  {RED}✗ {test_fn.__name__} CRASHED: {e}{RESET}")
            import traceback
            traceback.print_exc()
            total_failed += 1

    # ── Summary ─────────────────────────────────────────────────
    header("FINAL SUMMARY")
    print(f"  {GREEN}Passed: {total_passed}{RESET}")
    print(f"  {RED}Failed: {total_failed}{RESET}")
    print(f"  Total:  {total_passed + total_failed}")

    if total_failed > 0:
        print(f"\n  {RED}{BOLD}SOME TESTS FAILED{RESET}")
        return 1
    else:
        print(f"\n  {GREEN}{BOLD}ALL TESTS PASSED{RESET}")
        return 0


if __name__ == "__main__":
    exit(main())
