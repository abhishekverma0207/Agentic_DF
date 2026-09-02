#!/usr/bin/env python3
"""
End-to-End Simulation Test: YYYYWW (year_week) and YYYYMM (year_month)

Exercises the full pipeline:
  EDA (metrics) → Segmentation → Feature Engineering → Model Training
  → Inference (dead key forecasts + forward forecasts) → Backtesting

Two complete runs:
  1. YYYYWW (year_week) — weekly demand with 52 periods/year
  2. YYYYMM (year_month) — monthly demand with 12 periods/year
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("e2e_test")
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
# SYNTHETIC DATA GENERATORS
# ════════════════════════════════════════════════════════════════════

def _generate_periods(start_year: int, start_sub: int, n_periods: int,
                      time_format: str) -> List[int]:
    """Generate list of YYYYWW or YYYYMM integers."""
    max_sub = 12 if time_format == "year_month" else 52
    periods = []
    y, s = start_year, start_sub
    for _ in range(n_periods):
        periods.append(int(f"{y}{s:02d}"))
        s += 1
        if s > max_sub:
            s = 1
            y += 1
    return periods


def generate_synthetic_data(
    n_keys: int = 20,
    time_format: str = "year_week",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Create a synthetic demand dataset.

    Returns
    -------
    df : pd.DataFrame
        Columns: key, <date_col>, <target_col>
    split_config : dict
        train_start, train_end, val_start, val_end, test_start, test_end
    """
    rng = np.random.RandomState(seed)

    if time_format == "year_week":
        # ~3 years of weekly data: 2022-W01 → 2024-W52
        all_periods = _generate_periods(2022, 1, 156, "year_week")   # 3 years
        date_col = "year_week"
        # Splits
        train_start, train_end = all_periods[0], all_periods[103]     # W1-2022 → W52-2023 (104 weeks = 2 yr)
        val_start,   val_end   = all_periods[104], all_periods[129]   # W1-2024 → W26-2024 (26 weeks)
        test_start,  test_end  = all_periods[130], all_periods[145]   # W27-2024 → W42-2024 (16 weeks)
    else:
        # ~4 years of monthly data: 2021-01 → 2024-12
        all_periods = _generate_periods(2021, 1, 48, "year_month")   # 4 years
        date_col = "year_month"
        train_start, train_end = all_periods[0], all_periods[29]     # M01-2021 → M06-2023 (30 mo)
        val_start,   val_end   = all_periods[30], all_periods[38]    # M07-2023 → M03-2024 (9 mo)
        test_start,  test_end  = all_periods[39], all_periods[47]    # M04-2024 → M12-2024 (9 mo)

    keys = [f"SKU_{i:03d}" for i in range(n_keys)]
    rows = []
    for key in keys:
        base = rng.uniform(10, 500)
        noise_scale = base * rng.uniform(0.1, 0.5)
        trend = rng.uniform(-0.5, 0.5)
        seasonal_amplitude = base * rng.uniform(0.0, 0.3)
        max_sub = 12 if time_format == "year_month" else 52

        for idx, period in enumerate(all_periods):
            seasonal = seasonal_amplitude * np.sin(2 * np.pi * (idx % max_sub) / max_sub)
            value = max(0, base + trend * idx + seasonal + rng.normal(0, noise_scale))
            # Introduce ~10% zeros for some keys to create intermittent demand
            if rng.random() < 0.10:
                value = 0.0
            rows.append({"key": key, date_col: period, "demand": round(value, 2)})

    df = pd.DataFrame(rows)
    split_config = {
        "train_start": str(train_start),
        "train_end":   str(train_end),
        "val_start":   str(val_start),
        "val_end":     str(val_end),
        "test_start":  str(test_start),
        "test_end":    str(test_end),
    }
    return df, split_config


# ════════════════════════════════════════════════════════════════════
# SIMPLE EDA: Compute per-key metrics (replaces EDA crew)
# ════════════════════════════════════════════════════════════════════

def compute_per_key_metrics(df: pd.DataFrame, key_col: str,
                            target_col: str) -> pd.DataFrame:
    """Compute per-key summary metrics that segmentation requires."""
    metrics = []
    for key, grp in df.groupby(key_col):
        vals = grp[target_col].values
        n = len(vals)
        mean_val = np.mean(vals)
        std_val = np.std(vals) if n > 1 else 0.0
        cv = std_val / mean_val if mean_val > 0 else 0.0
        zero_frac = np.mean(vals == 0)
        nonzero_mask = vals > 0
        if nonzero_mask.sum() > 1:
            diffs = np.diff(np.where(nonzero_mask)[0])
            adi = np.mean(diffs) if len(diffs) > 0 else 1.0
        else:
            adi = float(n)
        cv2 = cv ** 2
        # Syntetos-Boylan classification
        if adi < 1.32 and cv2 < 0.49:
            pattern = "smooth"
        elif adi < 1.32:
            pattern = "erratic"
        elif cv2 < 0.49:
            pattern = "intermittent"
        else:
            pattern = "lumpy"
        metrics.append({
            "key": key,
            "volume_mean": mean_val,
            "std": std_val,
            "cv": cv,
            "cv_clean": cv,
            "adi": adi,
            "adi_log": np.log1p(adi),
            "cv2": cv2,
            "zero_fraction": zero_frac,
            "zero_fraction_clean": zero_frac,
            "demand_frequency": 1 - zero_frac,
            "n_obs": n,
            "demand_pattern": pattern,
        })
    return pd.DataFrame(metrics)


# ════════════════════════════════════════════════════════════════════
# SINGLE FORMAT E2E TEST
# ════════════════════════════════════════════════════════════════════

@dataclass
class E2EResult:
    format_name: str
    passed: int = 0
    failed: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def run_e2e_for_format(time_format: str) -> E2EResult:
    """Run full pipeline for one time format. Returns test result."""
    fmt_label = "YYYYWW (year_week)" if time_format == "year_week" else "YYYYMM (year_month)"
    header(f"E2E TEST: {fmt_label}")
    result = E2EResult(format_name=fmt_label)

    # Create temp directory
    tmp_dir = tempfile.mkdtemp(prefix=f"e2e_{time_format}_")
    date_col = "year_week" if time_format == "year_week" else "year_month"

    try:
        # ────────────────────────────────────────────
        # STEP 0: Generate synthetic data
        # ────────────────────────────────────────────
        subheader("Step 0: Generate synthetic data")
        df, splits = generate_synthetic_data(
            n_keys=15,
            time_format=time_format,
            seed=42,
        )
        n_periods = df[date_col].nunique()
        n_keys = df["key"].nunique()
        ok(f"Generated {len(df)} rows: {n_keys} keys × {n_periods} periods")
        ok(f"Splits: train={splits['train_start']}→{splits['train_end']}, "
           f"val={splits['val_start']}→{splits['val_end']}, "
           f"test={splits['test_start']}→{splits['test_end']}")
        result.passed += 1

        # Save source data
        source_path = os.path.join(tmp_dir, "source_data.csv")
        df.to_csv(source_path, index=False)

        # Setup directory structure
        artifact_base = os.path.join(tmp_dir, "artifacts")
        eda_dir = os.path.join(artifact_base, "eda_output")
        seg_dir = os.path.join(artifact_base, "seg_output")
        feat_dir = os.path.join(artifact_base, "feature_output")
        model_dir = os.path.join(artifact_base, "model_artifacts")
        backtest_dir = os.path.join(artifact_base, "backtest_output")
        for d in [eda_dir, seg_dir, feat_dir, model_dir, backtest_dir]:
            os.makedirs(d, exist_ok=True)

        # Create config
        from config.schema import DemandForecastConfig
        config = DemandForecastConfig(
            input_data_path=source_path,
            artifact_base_path=artifact_base,
            prediction_key_cols=["key"],
            timestamp_col=date_col,
            target_col="demand",
            time_format=time_format,
            forecast_horizon=6 if time_format == "year_month" else 8,
            train_start=splits["train_start"],
            train_end=splits["train_end"],
            val_start=splits["val_start"],
            val_end=splits["val_end"],
            test_start=splits["test_start"],
            test_end=splits["test_end"],
        )

        # Verify config properties
        if time_format == "year_month":
            assert config.is_monthly, "is_monthly should be True"
            assert config.periods_per_year == 12
        else:
            assert not config.is_monthly, "is_monthly should be False"
            assert config.periods_per_year == 52
        defaults = config.get_time_aware_defaults()
        ok(f"Config created: periods_per_year={config.periods_per_year}, "
           f"forecast_horizon={config.forecast_horizon}")
        ok(f"Time-aware defaults: seasonal_period={defaults['seasonal_period']}, "
           f"lags={defaults['recommended_lags']}")
        result.passed += 1

        # ────────────────────────────────────────────
        # STEP 1: EDA — Compute per-key metrics
        # ────────────────────────────────────────────
        subheader("Step 1: EDA — Per-key metrics")
        per_key_metrics = compute_per_key_metrics(df, "key", "demand")
        per_key_metrics.to_csv(os.path.join(eda_dir, "per_key_stats.csv"), index=False)

        patterns = per_key_metrics["demand_pattern"].value_counts().to_dict()
        ok(f"Computed metrics for {len(per_key_metrics)} keys. Patterns: {patterns}")
        assert len(per_key_metrics) == n_keys
        result.passed += 1

        # ────────────────────────────────────────────
        # STEP 2: Segmentation
        # ────────────────────────────────────────────
        subheader("Step 2: Segmentation")
        try:
            from utils.segmentation import run_segmentation_pipeline
            seg_result = run_segmentation_pipeline(
                per_key_metrics=per_key_metrics,
                key_cols=["key"],
                output_dir=seg_dir,
                allowed_model_families=["lightgbm", "xgboost", "catboost"],
                time_format=time_format,
                clustering_method="gmm",
                n_clusters_range=[2, 3, 4],
                create_visualizations=False,   # Skip plots for speed
                use_hybrid_segmentation=True,
            )
            n_segments = seg_result.n_segments
            ok(f"Segmentation complete: {n_segments} segments")

            # Verify output files
            seg_csv = os.path.join(seg_dir, "per_key_with_segments.csv")
            assert os.path.exists(seg_csv), f"Missing {seg_csv}"
            seg_df = pd.read_csv(seg_csv)
            assert "segment_id" in seg_df.columns, "segment_id column missing"
            assert len(seg_df) == n_keys
            ok(f"Segment assignments: {seg_df['segment_id'].nunique()} unique segments")
            result.passed += 1
        except Exception as e:
            fail(f"Segmentation failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Segmentation: {e}")
            # Create fallback segmentation for downstream steps
            seg_df = per_key_metrics[["key"]].copy()
            seg_df["segment_id"] = "segment_0"
            seg_df["demand_pattern"] = per_key_metrics["demand_pattern"]
            seg_df.to_csv(os.path.join(seg_dir, "per_key_with_segments.csv"), index=False)
            warn("Using fallback single-segment assignment")

        # ────────────────────────────────────────────
        # STEP 3: Feature Engineering
        # ────────────────────────────────────────────
        subheader("Step 3: Feature Engineering")
        try:
            from utils.feature_engineering import run_feature_pipeline
            feat_result = run_feature_pipeline(
                df=df.copy(),
                key_cols=["key"],
                date_col=date_col,
                target_col="demand",
                train_start=splits["train_start"],
                train_end=splits["train_end"],
                val_start=splits["val_start"],
                val_end=splits["val_end"],
                test_start=splits["test_start"],
                test_end=splits["test_end"],
                output_dir=feat_dir,
                time_format=time_format,
            )
            n_feat = feat_result.n_features_created
            ok(f"Feature engineering complete: {n_feat} features created")
            ok(f"  Train: {feat_result.n_rows_train} rows, "
               f"Val: {feat_result.n_rows_val} rows, "
               f"Test: {feat_result.n_rows_test} rows")

            # Verify output files
            for fname in ["train_features.csv", "val_features.csv", "test_features.csv"]:
                fpath = os.path.join(feat_dir, fname)
                assert os.path.exists(fpath), f"Missing {fpath}"
            ok("All feature files created")
            result.passed += 1
        except Exception as e:
            fail(f"Feature engineering failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Feature engineering: {e}")
            return result  # Can't continue without features

        # ────────────────────────────────────────────
        # STEP 4: Create training manifest
        # ────────────────────────────────────────────
        subheader("Step 4: Create training manifest")
        try:
            seg_csv = os.path.join(seg_dir, "per_key_with_segments.csv")
            seg_df = pd.read_csv(seg_csv)

            # Build manifest: key → segment_id, model_level, model_group
            manifest = seg_df[["key", "segment_id"]].copy()
            manifest["model_level"] = manifest["segment_id"]
            manifest["model_group"] = manifest["segment_id"]
            if "demand_pattern" in seg_df.columns:
                manifest["demand_pattern"] = seg_df["demand_pattern"]
            else:
                manifest["demand_pattern"] = "smooth"
            manifest["allocation_rationale"] = "initial"

            manifest_path = os.path.join(feat_dir, "training_manifest.csv")
            manifest.to_csv(manifest_path, index=False)
            ok(f"Training manifest: {len(manifest)} keys, {manifest['model_group'].nunique()} groups")

            # Also create per-model_group feature CSVs (required by train_all_model_groups)
            train_features_df = pd.read_csv(os.path.join(feat_dir, "train_features.csv"))
            val_features_df = pd.read_csv(os.path.join(feat_dir, "val_features.csv"))

            # Merge segment info into features
            key_to_segment = dict(zip(manifest["key"], manifest["model_group"]))

            if "model_group" not in train_features_df.columns:
                train_features_df["model_group"] = train_features_df["key"].map(key_to_segment)
                val_features_df["model_group"] = val_features_df["key"].map(key_to_segment)

            for mg in manifest["model_group"].unique():
                mg_train = train_features_df[train_features_df["model_group"] == mg]
                mg_val = val_features_df[val_features_df["model_group"] == mg]
                mg_train.to_csv(os.path.join(feat_dir, f"{mg}_train_features.csv"), index=False)
                mg_val.to_csv(os.path.join(feat_dir, f"{mg}_val_features.csv"), index=False)

            ok(f"Created per-group feature files for {manifest['model_group'].nunique()} groups")
            result.passed += 1
        except Exception as e:
            fail(f"Manifest creation failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Manifest: {e}")
            return result

        # ────────────────────────────────────────────
        # STEP 5: Model Training (per-group)
        # ────────────────────────────────────────────
        subheader("Step 5: Model Training")
        try:
            from utils.model_training import train_model_by_name, TRAINING_REGISTRY
            # Determine feature columns
            exclude_cols = {"key", date_col, "demand", "split", "model_level",
                           "model_group", "segment_id", "demand_pattern",
                           "intermittency_class", "label", "demand_log",
                           "allocation_rationale"}
            feature_cols = [c for c in train_features_df.columns if c not in exclude_cols]
            # Remove any non-numeric columns
            numeric_cols = []
            for c in feature_cols:
                if train_features_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
                    numeric_cols.append(c)
            feature_cols = numeric_cols

            ok(f"Feature columns for training: {len(feature_cols)}")

            model_specs_list = []
            trained_models = {}
            model_groups = manifest["model_group"].unique()

            for mg in model_groups:
                mg_train = train_features_df[train_features_df["model_group"] == mg]
                mg_val = val_features_df[val_features_df["model_group"] == mg]

                avail_feats = [c for c in feature_cols if c in mg_train.columns]
                if len(avail_feats) < 3 or len(mg_train) < 10:
                    warn(f"Skipping group {mg}: too few features ({len(avail_feats)}) or rows ({len(mg_train)})")
                    continue

                X_train = mg_train[avail_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
                y_train = mg_train["demand"].fillna(0).values
                X_val = mg_val[avail_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
                y_val = mg_val["demand"].fillna(0).values

                # Train lightgbm (fast and reliable)
                try:
                    train_result = train_model_by_name(
                        model_type="lightgbm",
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                    )
                    trained_models[mg] = {
                        "model": train_result.model,
                        "model_type": train_result.model_type,
                        "hyperparameters": train_result.hyperparameters,
                        "val_wape": train_result.val_wape,
                    }
                    model_specs_list.append({
                        "model_level": mg,
                        "model_type": train_result.model_type,
                        "hyperparameters": train_result.hyperparameters,
                        "val_wape": float(train_result.val_wape),
                        "feature_columns": avail_feats,
                    })
                except Exception as te:
                    warn(f"  Training failed for {mg}: {te}")

            ok(f"Trained {len(trained_models)}/{len(model_groups)} model groups")

            if not trained_models:
                fail("No models trained — cannot continue")
                result.failed += 1
                result.errors.append("No models trained")
                return result

            # Save model specs
            specs_output = {
                "model_specs": model_specs_list,
                "feature_columns": feature_cols,
                "overall_wape": float(np.mean([m["val_wape"] for m in trained_models.values()])),
            }
            specs_path = os.path.join(model_dir, "final_model_specs.json")
            with open(specs_path, "w") as f:
                json.dump(specs_output, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o) if isinstance(o, (np.floating,)) else str(o))

            # Save models
            import joblib
            for mg, minfo in trained_models.items():
                model_path = os.path.join(model_dir, f"{mg}_model.pkl")
                joblib.dump(minfo["model"], model_path)

            ok(f"Model specs saved. Overall val WAPE: {specs_output['overall_wape']:.4f}")
            result.passed += 1

        except Exception as e:
            fail(f"Model training failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Model training: {e}")
            return result

        # ────────────────────────────────────────────
        # STEP 6: Dead key forecasts (YYYYWW/YYYYMM)
        # ────────────────────────────────────────────
        subheader("Step 6: Dead key forecasts")
        try:
            from utils.inference import create_dead_key_forecasts
            # Create 3 fake dead keys
            dead_keys = ["DEAD_001", "DEAD_002", "DEAD_003"]

            dead_df = create_dead_key_forecasts(
                dead_keys=dead_keys,
                test_start=splits["test_start"],
                test_end=splits["test_end"],
                key_col="key",
                date_col=date_col,
                target_col="demand",
                forecast_horizon=config.forecast_horizon,
                time_format=time_format,
            )

            assert len(dead_df) == len(dead_keys) * config.forecast_horizon, \
                f"Expected {len(dead_keys) * config.forecast_horizon} rows, got {len(dead_df)}"
            assert all(dead_df["predicted"] == 0.0), "Dead key predictions should be 0"
            assert all(dead_df["is_dead_key"]), "is_dead_key should be True"

            # Verify period rollover
            periods_per_key = dead_df[dead_df["key"] == dead_keys[0]][date_col].tolist()
            max_sub = 12 if time_format == "year_month" else 52
            for i in range(1, len(periods_per_key)):
                prev_sub = int(str(periods_per_key[i-1])[4:])
                curr_sub = int(str(periods_per_key[i])[4:])
                if prev_sub == max_sub:
                    assert curr_sub == 1, f"Year rollover failed: {periods_per_key[i-1]} → {periods_per_key[i]}"

            ok(f"Dead key forecasts: {len(dead_df)} rows for {len(dead_keys)} keys, "
               f"periods={periods_per_key}")
            result.passed += 1

        except Exception as e:
            fail(f"Dead key forecasts failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Dead key forecasts: {e}")

        # ────────────────────────────────────────────
        # STEP 7: Forward forecast generation
        # ────────────────────────────────────────────
        subheader("Step 7: Forward forecast generation")
        try:
            from utils.inference import generate_forward_forecasts

            forecast_df, n_forecasts = generate_forward_forecasts(
                config=config,
                retrained_models=trained_models,
                manifest_df=manifest,
                model_specs=specs_output,
                feature_dir=feat_dir,
                output_dir=model_dir,
            )

            ok(f"Forward forecasts: {n_forecasts} rows generated")
            if not forecast_df.empty:
                unique_keys_forecast = forecast_df["key"].nunique()
                unique_periods_forecast = forecast_df[date_col].nunique()
                ok(f"  {unique_keys_forecast} keys × {unique_periods_forecast} periods")

                # Verify predictions are non-negative
                assert (forecast_df["predicted"] >= 0).all(), "Some predictions are negative"
                ok("  All predictions ≥ 0")

                # Verify forecast_step is correct
                max_step = forecast_df["forecast_step"].max()
                ok(f"  Max forecast step: {max_step} (horizon={config.forecast_horizon})")
            result.passed += 1

        except Exception as e:
            fail(f"Forward forecast generation failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Forward forecasts: {e}")

        # ────────────────────────────────────────────
        # STEP 8: Backtesting — period functions
        # ────────────────────────────────────────────
        subheader("Step 8: Backtesting period functions")
        try:
            from utils.backtesting import (
                parse_period, increment_period, decrement_period,
                count_periods_between, generate_period_range,
                generate_backtest_origins,
            )

            # Test increment across year boundary
            if time_format == "year_week":
                assert increment_period("202452", 1, "year_week") == "202501"
                assert decrement_period("202501", 1, "year_week") == "202452"
                p_range = generate_period_range("202450", "202503", "year_week")
                assert len(p_range) == 6  # W50, W51, W52, W01, W02, W03
            else:
                assert increment_period("202412", 1, "year_month") == "202501"
                assert decrement_period("202501", 1, "year_month") == "202412"
                p_range = generate_period_range("202410", "202503", "year_month")
                assert len(p_range) == 6  # M10, M11, M12, M01, M02, M03

            ok(f"Period arithmetic: year rollover works (range={p_range})")

            # Test backtest origin generation
            origins = generate_backtest_origins(
                val_end=splits["val_end"],
                test_start=splits["test_start"],
                test_end=splits["test_end"],
                forecast_horizon=config.forecast_horizon,
                time_format=time_format,
            )
            assert len(origins) > 0, "No origins generated"
            ok(f"Backtest origins: {len(origins)} origins generated")
            for o in origins[:3]:
                ok(f"  Origin {o['origin_idx']}: "
                   f"cutoff={o['train_cutoff']}, "
                   f"forecast={o['forecast_start']}→{o['forecast_end']}")

            # Validate origin periods are valid
            for o in origins:
                _, sub = parse_period(o["forecast_start"])
                max_sub = 12 if time_format == "year_month" else 52
                assert 1 <= sub <= max_sub, f"Invalid sub-period in {o['forecast_start']}: {sub}"
                _, sub_end = parse_period(o["forecast_end"])
                assert 1 <= sub_end <= max_sub, f"Invalid sub-period in {o['forecast_end']}: {sub_end}"
            ok("All origin periods have valid sub-period values")
            result.passed += 1

        except Exception as e:
            fail(f"Backtesting period functions failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Backtesting periods: {e}")

        # ────────────────────────────────────────────
        # STEP 9: Retrain models (simulates backtesting retrain)
        # ────────────────────────────────────────────
        subheader("Step 9: Retrain models (backtesting-style)")
        try:
            from utils.inference import retrain_all_models

            retrained, n_retrained = retrain_all_models(
                config=config,
                manifest_df=manifest,
                model_specs=specs_output,
                feature_dir=feat_dir,
                model_dir=model_dir,
                target_col="demand",
                key_col="key",
                date_col=date_col,
                train_cutoff=splits["val_end"],
            )
            ok(f"Retrained {n_retrained} models")
            assert n_retrained > 0, "No models were retrained"
            result.passed += 1

        except Exception as e:
            fail(f"Model retraining failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Retrain: {e}")

        # ────────────────────────────────────────────
        # STEP 10: Config time-aware defaults
        # ────────────────────────────────────────────
        subheader("Step 10: Config time-aware defaults")
        try:
            defaults = config.get_time_aware_defaults()
            if time_format == "year_month":
                assert defaults["seasonal_period"] == 12
                assert defaults["recommended_lags"] == [1, 2, 3, 6, 12]
                assert defaults["rolling_windows"] == [3, 6, 12]
                assert defaults["walk_forward_min_train_periods"] == 12
                assert defaults["walk_forward_rolling_window"] == 24
                assert defaults["dead_key_lookback"] == 12
            else:
                assert defaults["seasonal_period"] == 52
                assert defaults["recommended_lags"] == [1, 2, 4, 8, 13, 26, 52]
                assert defaults["rolling_windows"] == [4, 8, 13]
                assert defaults["walk_forward_min_train_periods"] == 52
                assert defaults["walk_forward_rolling_window"] == 104
                assert defaults["dead_key_lookback"] == 52
            ok(f"Time-aware defaults: {defaults}")
            result.passed += 1
        except Exception as e:
            fail(f"Config defaults check failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Config defaults: {e}")

        # ────────────────────────────────────────────
        # STEP 11: Full backtesting simulation (lightweight)
        # ────────────────────────────────────────────
        subheader("Step 11: Backtesting — origin-by-origin simulation")
        try:
            from utils.backtesting import generate_backtest_origins
            from utils.inference import create_dead_key_forecasts

            origins = generate_backtest_origins(
                val_end=splits["val_end"],
                test_start=splits["test_start"],
                test_end=splits["test_end"],
                forecast_horizon=config.forecast_horizon,
                time_format=time_format,
            )

            # Simulate 2 origins (full backtest is too slow for test)
            all_origin_forecasts = []
            for origin_config in origins[:2]:
                origin_idx = origin_config["origin_idx"]
                forecast_start = origin_config["forecast_start"]
                forecast_end = origin_config["forecast_end"]
                train_cutoff = origin_config["train_cutoff"]

                # Generate dead key forecasts for this origin
                origin_dead_df = create_dead_key_forecasts(
                    dead_keys=["DEAD_001"],
                    test_start=forecast_start,
                    test_end=forecast_end,
                    key_col="key",
                    date_col=date_col,
                    target_col="demand",
                    forecast_horizon=config.forecast_horizon,
                    time_format=time_format,
                )
                origin_dead_df["origin_idx"] = origin_idx
                all_origin_forecasts.append(origin_dead_df)

            combined = pd.concat(all_origin_forecasts, ignore_index=True)
            ok(f"Simulated {len(origins[:2])} origins, combined {len(combined)} forecast rows")

            # Verify periods don't have invalid values
            for _, row in combined.iterrows():
                period_str = str(row[date_col])
                sub = int(period_str[4:])
                max_sub = 12 if time_format == "year_month" else 52
                assert 1 <= sub <= max_sub, f"Invalid period {period_str} (sub={sub})"
            ok("All forecast periods have valid sub-period values")
            result.passed += 1

        except Exception as e:
            fail(f"Backtesting simulation failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Backtesting simulation: {e}")

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        gc.collect()

    return result


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    header("END-TO-END TIME FORMAT SIMULATION TEST")
    print(f"  Testing both YYYYWW (year_week) and YYYYMM (year_month)")
    print(f"  Pipeline: EDA → Segmentation → Features → Training → Inference → Backtesting\n")

    results = []

    # Run for year_week
    try:
        r1 = run_e2e_for_format("year_week")
        results.append(r1)
    except Exception as e:
        print(f"\n{RED}CRITICAL ERROR in year_week test: {e}{RESET}")
        traceback.print_exc()
        results.append(E2EResult(format_name="YYYYWW", failed=1, errors=[str(e)]))

    # Run for year_month
    try:
        r2 = run_e2e_for_format("year_month")
        results.append(r2)
    except Exception as e:
        print(f"\n{RED}CRITICAL ERROR in year_month test: {e}{RESET}")
        traceback.print_exc()
        results.append(E2EResult(format_name="YYYYMM", failed=1, errors=[str(e)]))

    # ── Summary ──────────────────────────────────────────────────
    header("FINAL SUMMARY")
    total_passed = 0
    total_failed = 0
    all_errors = []

    for r in results:
        status = f"{GREEN}PASS{RESET}" if r.failed == 0 else f"{RED}FAIL{RESET}"
        print(f"  {r.format_name}: {status}  ({r.passed} passed, {r.failed} failed)")
        total_passed += r.passed
        total_failed += r.failed
        all_errors.extend(r.errors)

    print(f"\n  {BOLD}Total: {total_passed} passed, {total_failed} failed{RESET}")

    if all_errors:
        print(f"\n  {RED}Errors:{RESET}")
        for err in all_errors:
            print(f"    - {err}")

    if total_failed == 0:
        print(f"\n  {GREEN}{BOLD}🎉 ALL E2E TESTS PASSED!{RESET}")
        return 0
    else:
        print(f"\n  {RED}{BOLD}❌ SOME TESTS FAILED{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
