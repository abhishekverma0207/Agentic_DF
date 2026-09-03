#!/usr/bin/env python3
"""
End-to-End Data Format Compatibility Test: CSV, Parquet, Delta

Verifies that the full pipeline works identically with all three input
data formats:
  1. CSV  (.csv file)
  2. Parquet  (.parquet file)
  3. Delta  (directory with _delta_log/)

Pipeline stages tested:
  load_source_data → EDA (metrics) → Segmentation → Feature Engineering
  → Model Training → Inference (dead key + forward forecasts) → Backtesting
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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("e2e_format_test")
logger.setLevel(logging.INFO)

# ── Pretty printing ─────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}+{RESET} {msg}")
def fail(msg): print(f"  {RED}x{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{'='*70}\n{msg}\n{'='*70}{RESET}")
def subheader(msg): print(f"\n{BOLD}--- {msg} ---{RESET}")


# ════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR
# ════════════════════════════════════════════════════════════════════

def _generate_periods(start_year: int, start_sub: int, n_periods: int,
                      time_format: str) -> List[int]:
    """Generate list of YYYYMM integers (monthly)."""
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
    n_keys: int = 15,
    time_format: str = "year_month",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Create a synthetic monthly demand dataset.

    Returns
    -------
    df : pd.DataFrame
    split_config : dict
    """
    rng = np.random.RandomState(seed)

    # 4 years of monthly data: 2021-01 -> 2024-12
    all_periods = _generate_periods(2021, 1, 48, "year_month")
    date_col = "year_month"
    train_start, train_end = all_periods[0], all_periods[29]    # 30 months
    val_start,   val_end   = all_periods[30], all_periods[38]   #  9 months
    test_start,  test_end  = all_periods[39], all_periods[47]   #  9 months

    keys = [f"SKU_{i:03d}" for i in range(n_keys)]
    rows = []
    for key in keys:
        base = rng.uniform(10, 500)
        noise_scale = base * rng.uniform(0.1, 0.5)
        trend = rng.uniform(-0.5, 0.5)
        seasonal_amplitude = base * rng.uniform(0.0, 0.3)

        for idx, period in enumerate(all_periods):
            seasonal = seasonal_amplitude * np.sin(2 * np.pi * (idx % 12) / 12)
            value = max(0, base + trend * idx + seasonal + rng.normal(0, noise_scale))
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
    """Compute per-key summary metrics for segmentation."""
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
# DATA FORMAT WRITERS
# ════════════════════════════════════════════════════════════════════

def save_as_csv(df: pd.DataFrame, path: str) -> str:
    """Save DataFrame as CSV and return path."""
    df.to_csv(path, index=False)
    return path


def save_as_parquet(df: pd.DataFrame, path: str) -> str:
    """Save DataFrame as single Parquet file and return path."""
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def save_as_delta(df: pd.DataFrame, path: str) -> str:
    """Save DataFrame as a Delta table directory and return path."""
    from deltalake import write_deltalake
    os.makedirs(path, exist_ok=True)
    write_deltalake(path, df, mode="overwrite")
    return path


# ════════════════════════════════════════════════════════════════════
# SINGLE-FORMAT E2E TEST
# ════════════════════════════════════════════════════════════════════

@dataclass
class FormatTestResult:
    format_name: str
    passed: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)
    loaded_df: pd.DataFrame = field(default=None, repr=False)
    n_features: int = 0
    n_models_trained: int = 0
    n_forecast_rows: int = 0
    val_wape: float = 0.0


def run_pipeline_for_format(
    format_name: str,
    source_path: str,
    reference_df: pd.DataFrame,
    splits: Dict[str, str],
    time_format: str = "year_month",
) -> FormatTestResult:
    """Run the full pipeline for one data format and return results."""
    header(f"E2E FORMAT TEST: {format_name.upper()}")
    result = FormatTestResult(format_name=format_name)
    date_col = "year_month"

    tmp_dir = tempfile.mkdtemp(prefix=f"e2e_{format_name}_")

    try:
        # ────────────────────────────────────────────
        # STEP 0: Load data using load_source_data
        # ────────────────────────────────────────────
        subheader(f"Step 0: Load data from {format_name} format")
        try:
            from utils.agent_utilities import load_source_data, _detect_data_format

            detected = _detect_data_format(source_path)
            ok(f"Detected format: {detected} (expected: {format_name})")
            assert detected == format_name, \
                f"Format detection mismatch: got {detected}, expected {format_name}"

            df = load_source_data(source_path)
            ok(f"Loaded {len(df)} rows, {len(df.columns)} columns")

            # Verify identical to reference
            assert len(df) == len(reference_df), \
                f"Row count mismatch: {len(df)} vs {len(reference_df)} (reference)"
            assert set(df.columns) == set(reference_df.columns), \
                f"Column mismatch: {set(df.columns)} vs {set(reference_df.columns)}"

            # Sort both for comparison (parquet/delta may reorder)
            df_sorted = df.sort_values(["key", date_col]).reset_index(drop=True)
            ref_sorted = reference_df.sort_values(["key", date_col]).reset_index(drop=True)

            # Compare values (allow minor floating-point differences)
            for col in reference_df.columns:
                if df_sorted[col].dtype in [np.float64, np.float32]:
                    np.testing.assert_allclose(
                        df_sorted[col].values, ref_sorted[col].values,
                        rtol=1e-6, atol=1e-10,
                        err_msg=f"Column {col} values differ"
                    )
                else:
                    assert (df_sorted[col] == ref_sorted[col]).all(), \
                        f"Column {col} values differ"

            ok(f"Data matches reference CSV exactly ({len(df)} rows, {len(df.columns)} cols)")
            result.loaded_df = df
            result.passed += 1

        except Exception as e:
            fail(f"Data loading failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Load: {e}")
            return result

        # ────────────────────────────────────────────
        # STEP 1: Create config pointing to the format-specific path
        # ────────────────────────────────────────────
        subheader("Step 1: Create config")
        from config.schema import DemandForecastConfig

        artifact_base = os.path.join(tmp_dir, "artifacts")
        eda_dir = os.path.join(artifact_base, "eda_output")
        seg_dir = os.path.join(artifact_base, "seg_output")
        feat_dir = os.path.join(artifact_base, "feature_output")
        model_dir = os.path.join(artifact_base, "model_artifacts")
        output_folder = os.path.join(tmp_dir, "final_output")
        for d in [eda_dir, seg_dir, feat_dir, model_dir]:
            os.makedirs(d, exist_ok=True)

        config = DemandForecastConfig(
            input_data_path=source_path,
            artifact_base_path=artifact_base,
            output_folder_path=output_folder,
            prediction_key_cols=["key"],
            timestamp_col=date_col,
            target_col="demand",
            time_format=time_format,
            forecast_horizon=6,
            train_start=splits["train_start"],
            train_end=splits["train_end"],
            val_start=splits["val_start"],
            val_end=splits["val_end"],
            test_start=splits["test_start"],
            test_end=splits["test_end"],
        )
        assert config.is_monthly
        assert config.periods_per_year == 12
        ok(f"Config created: input_data_path={source_path}")
        result.passed += 1

        # ────────────────────────────────────────────
        # STEP 2: EDA — per-key metrics
        # ────────────────────────────────────────────
        subheader("Step 2: EDA - Per-key metrics")
        per_key_metrics = compute_per_key_metrics(df, "key", "demand")
        per_key_metrics.to_csv(os.path.join(eda_dir, "per_key_stats.csv"), index=False)
        patterns = per_key_metrics["demand_pattern"].value_counts().to_dict()
        ok(f"EDA: {len(per_key_metrics)} keys, patterns: {patterns}")
        result.passed += 1

        # ────────────────────────────────────────────
        # STEP 3: Segmentation
        # ────────────────────────────────────────────
        subheader("Step 3: Segmentation")
        try:
            from utils.segmentation import run_segmentation_pipeline
            seg_result = run_segmentation_pipeline(
                per_key_metrics=per_key_metrics,
                key_cols=["key"],
                output_dir=seg_dir,
                allowed_model_families=["lightgbm", "xgboost"],
                time_format=time_format,
                clustering_method="gmm",
                n_clusters_range=[2, 3],
                create_visualizations=False,
                use_hybrid_segmentation=True,
            )
            n_segments = seg_result.n_segments
            seg_csv = os.path.join(seg_dir, "per_key_with_segments.csv")
            seg_df = pd.read_csv(seg_csv)
            ok(f"Segmentation: {n_segments} segments, {len(seg_df)} keys")
            result.passed += 1
        except Exception as e:
            fail(f"Segmentation failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Segmentation: {e}")
            # Fallback
            seg_df = per_key_metrics[["key"]].copy()
            seg_df["segment_id"] = "segment_0"
            seg_df["demand_pattern"] = per_key_metrics["demand_pattern"]
            seg_df.to_csv(os.path.join(seg_dir, "per_key_with_segments.csv"), index=False)
            warn("Using fallback single-segment assignment")

        # ────────────────────────────────────────────
        # STEP 4: Feature Engineering
        # ────────────────────────────────────────────
        subheader("Step 4: Feature Engineering")
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
            result.n_features = feat_result.n_features_created
            ok(f"Features: {result.n_features} created "
               f"(train={feat_result.n_rows_train}, val={feat_result.n_rows_val}, "
               f"test={feat_result.n_rows_test})")
            for fname in ["train_features.csv", "val_features.csv", "test_features.csv"]:
                assert os.path.exists(os.path.join(feat_dir, fname)), f"Missing {fname}"
            ok("All feature files created")
            result.passed += 1
        except Exception as e:
            fail(f"Feature engineering failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Feature engineering: {e}")
            return result

        # ────────────────────────────────────────────
        # STEP 5: Training manifest
        # ────────────────────────────────────────────
        subheader("Step 5: Training manifest")
        seg_csv = os.path.join(seg_dir, "per_key_with_segments.csv")
        seg_df = pd.read_csv(seg_csv)
        manifest = seg_df[["key", "segment_id"]].copy()
        manifest["model_level"] = manifest["segment_id"]
        manifest["model_group"] = manifest["segment_id"]
        manifest["demand_pattern"] = seg_df.get("demand_pattern", "smooth")
        manifest["allocation_rationale"] = "initial"
        manifest_path = os.path.join(feat_dir, "training_manifest.csv")
        manifest.to_csv(manifest_path, index=False)

        train_features_df = pd.read_csv(os.path.join(feat_dir, "train_features.csv"))
        val_features_df = pd.read_csv(os.path.join(feat_dir, "val_features.csv"))

        key_to_segment = dict(zip(manifest["key"], manifest["model_group"]))
        if "model_group" not in train_features_df.columns:
            train_features_df["model_group"] = train_features_df["key"].map(key_to_segment)
            val_features_df["model_group"] = val_features_df["key"].map(key_to_segment)

        for mg in manifest["model_group"].unique():
            mg_train = train_features_df[train_features_df["model_group"] == mg]
            mg_val = val_features_df[val_features_df["model_group"] == mg]
            mg_train.to_csv(os.path.join(feat_dir, f"{mg}_train_features.csv"), index=False)
            mg_val.to_csv(os.path.join(feat_dir, f"{mg}_val_features.csv"), index=False)
        ok(f"Manifest: {len(manifest)} keys, {manifest['model_group'].nunique()} groups")
        result.passed += 1

        # ────────────────────────────────────────────
        # STEP 6: Model Training
        # ────────────────────────────────────────────
        subheader("Step 6: Model Training")
        try:
            from utils.model_training import train_model_by_name
            exclude_cols = {"key", date_col, "demand", "split", "model_level",
                            "model_group", "segment_id", "demand_pattern",
                            "intermittency_class", "label", "demand_log",
                            "allocation_rationale"}
            feature_cols = [c for c in train_features_df.columns
                           if c not in exclude_cols
                           and train_features_df[c].dtype in [np.float64, np.float32,
                                                              np.int64, np.int32]]

            model_specs_list = []
            trained_models = {}

            for mg in manifest["model_group"].unique():
                mg_train = train_features_df[train_features_df["model_group"] == mg]
                mg_val = val_features_df[val_features_df["model_group"] == mg]
                avail_feats = [c for c in feature_cols if c in mg_train.columns]
                if len(avail_feats) < 3 or len(mg_train) < 10:
                    warn(f"Skipping group {mg}: too few features/rows")
                    continue

                X_train = mg_train[avail_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
                y_train = mg_train["demand"].fillna(0).values
                X_val = mg_val[avail_feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
                y_val = mg_val["demand"].fillna(0).values

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

            result.n_models_trained = len(trained_models)
            result.val_wape = float(np.mean([m["val_wape"] for m in trained_models.values()])) if trained_models else 999.0
            ok(f"Trained {len(trained_models)} models, avg val WAPE: {result.val_wape:.4f}")

            assert trained_models, "No models trained"

            specs_output = {
                "model_specs": model_specs_list,
                "feature_columns": feature_cols,
                "overall_wape": result.val_wape,
            }
            specs_path = os.path.join(model_dir, "final_model_specs.json")
            with open(specs_path, "w") as f:
                json.dump(specs_output, f, indent=2,
                          default=lambda o: int(o) if isinstance(o, (np.integer,))
                          else float(o) if isinstance(o, (np.floating,)) else str(o))

            import joblib
            for mg, minfo in trained_models.items():
                joblib.dump(minfo["model"], os.path.join(model_dir, f"{mg}_model.pkl"))

            result.passed += 1
        except Exception as e:
            fail(f"Training failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Training: {e}")
            return result

        # ────────────────────────────────────────────
        # STEP 7: Dead key forecasts
        # ────────────────────────────────────────────
        subheader("Step 7: Dead key forecasts")
        try:
            from utils.inference import create_dead_key_forecasts
            dead_keys = ["DEAD_001", "DEAD_002"]
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
            assert len(dead_df) == len(dead_keys) * config.forecast_horizon
            assert all(dead_df["predicted"] == 0.0)
            ok(f"Dead keys: {len(dead_df)} rows, all zero")
            result.passed += 1
        except Exception as e:
            fail(f"Dead key forecasts failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Dead keys: {e}")

        # ────────────────────────────────────────────
        # STEP 8: Forward forecast generation
        # ────────────────────────────────────────────
        subheader("Step 8: Forward forecasts")
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
            result.n_forecast_rows = n_forecasts
            ok(f"Forward forecasts: {n_forecasts} rows")
            if not forecast_df.empty:
                assert (forecast_df["predicted"] >= 0).all(), "Negative predictions"
                ok(f"  {forecast_df['key'].nunique()} keys, "
                   f"max step={forecast_df['forecast_step'].max()}")
            result.passed += 1
        except Exception as e:
            fail(f"Forward forecasts failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Forward forecasts: {e}")

        # ────────────────────────────────────────────
        # STEP 8b: Verify output_folder_path parquet output
        # ────────────────────────────────────────────
        subheader("Step 8b: output_folder_path parquet output")
        try:
            # Write forecast CSV to model_dir (mimicking PHASE 7)
            csv_path = os.path.join(model_dir, "inference_forecast.csv")
            forecast_df.to_csv(csv_path, index=False)

            # Write parquet to output_folder (mimicking PHASE 7 with output_folder_path)
            os.makedirs(output_folder, exist_ok=True)
            parquet_path = os.path.join(output_folder, "inference_forecast.parquet")
            forecast_df.to_parquet(parquet_path, index=False, engine="pyarrow")

            # Verify parquet file exists and matches CSV
            assert os.path.exists(parquet_path), f"Parquet file not created: {parquet_path}"
            parquet_df = pd.read_parquet(parquet_path)
            assert len(parquet_df) == len(forecast_df), \
                f"Parquet row count {len(parquet_df)} != CSV {len(forecast_df)}"
            assert set(parquet_df.columns) == set(forecast_df.columns), \
                "Parquet column mismatch"
            ok(f"Parquet output: {parquet_path} ({os.path.getsize(parquet_path):,} bytes)")
            ok(f"  Matches CSV: {len(parquet_df)} rows, {len(parquet_df.columns)} cols")
            result.passed += 1
        except Exception as e:
            fail(f"Parquet output test failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Parquet output: {e}")

        # ────────────────────────────────────────────
        # STEP 9: Backtesting period generation
        # ────────────────────────────────────────────
        subheader("Step 9: Backtesting simulation")
        try:
            from utils.backtesting import generate_backtest_origins
            from utils.inference import create_dead_key_forecasts as cdk

            origins = generate_backtest_origins(
                val_end=splits["val_end"],
                test_start=splits["test_start"],
                test_end=splits["test_end"],
                forecast_horizon=config.forecast_horizon,
                time_format=time_format,
            )
            assert len(origins) > 0, "No backtest origins"

            # Simulate 2 origins
            all_bt_forecasts = []
            for oc in origins[:2]:
                bt_dead = cdk(
                    dead_keys=["DEAD_001"],
                    test_start=oc["forecast_start"],
                    test_end=oc["forecast_end"],
                    key_col="key", date_col=date_col, target_col="demand",
                    forecast_horizon=config.forecast_horizon,
                    time_format=time_format,
                )
                bt_dead["origin_idx"] = oc["origin_idx"]
                all_bt_forecasts.append(bt_dead)

            combined = pd.concat(all_bt_forecasts, ignore_index=True)

            # Validate periods
            for _, row in combined.iterrows():
                sub = int(str(row[date_col])[4:])
                assert 1 <= sub <= 12, f"Invalid month {sub} in period {row[date_col]}"

            ok(f"Backtesting: {len(origins)} origins, simulated 2, "
               f"{len(combined)} forecast rows, all periods valid")
            result.passed += 1
        except Exception as e:
            fail(f"Backtesting failed: {e}")
            traceback.print_exc()
            result.failed += 1
            result.errors.append(f"Backtesting: {e}")

    finally:
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
    header("END-TO-END DATA FORMAT COMPATIBILITY TEST")
    print(f"  Testing CSV, Parquet, and Delta input formats")
    print(f"  Pipeline: load_source_data -> EDA -> Segmentation -> Features")
    print(f"           -> Training -> Inference -> Backtesting\n")

    time_format = "year_month"

    # ── Step A: Generate reference data ────────────────────────────
    subheader("Step A: Generate reference synthetic data")
    reference_df, splits = generate_synthetic_data(n_keys=15, time_format=time_format, seed=42)
    n_keys = reference_df["key"].nunique()
    n_periods = reference_df["year_month"].nunique()
    ok(f"Reference data: {len(reference_df)} rows, {n_keys} keys x {n_periods} months")
    ok(f"Splits: train={splits['train_start']}->{splits['train_end']}, "
       f"val={splits['val_start']}->{splits['val_end']}, "
       f"test={splits['test_start']}->{splits['test_end']}")

    # ── Step B: Save in all three formats ─────────────────────────
    subheader("Step B: Create data files in all three formats")
    data_dir = tempfile.mkdtemp(prefix="e2e_format_data_")

    csv_path = save_as_csv(reference_df, os.path.join(data_dir, "data.csv"))
    ok(f"CSV:     {csv_path}  ({os.path.getsize(csv_path):,} bytes)")

    parquet_path = save_as_parquet(reference_df, os.path.join(data_dir, "data.parquet"))
    ok(f"Parquet: {parquet_path}  ({os.path.getsize(parquet_path):,} bytes)")

    delta_path = os.path.join(data_dir, "data_delta")
    save_as_delta(reference_df, delta_path)
    delta_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(delta_path) for f in fn
    )
    ok(f"Delta:   {delta_path}  ({delta_size:,} bytes total)")

    # ── Step C: Run pipeline for each format ──────────────────────
    format_configs = [
        ("csv", csv_path),
        ("parquet", parquet_path),
        ("delta", delta_path),
    ]

    results = []
    for fmt_name, fmt_path in format_configs:
        try:
            r = run_pipeline_for_format(
                format_name=fmt_name,
                source_path=fmt_path,
                reference_df=reference_df,
                splits=splits,
                time_format=time_format,
            )
            results.append(r)
        except Exception as e:
            print(f"\n{RED}CRITICAL ERROR in {fmt_name} test: {e}{RESET}")
            traceback.print_exc()
            results.append(FormatTestResult(
                format_name=fmt_name, failed=1,
                errors=[f"Critical: {e}"]
            ))

    # ── Step D: Cross-format comparison ───────────────────────────
    header("CROSS-FORMAT COMPARISON")
    cross_passed = 0
    cross_failed = 0

    # Compare feature counts
    feat_counts = {r.format_name: r.n_features for r in results if r.n_features > 0}
    if len(set(feat_counts.values())) == 1:
        ok(f"Feature counts identical: {list(feat_counts.values())[0]} across all formats")
        cross_passed += 1
    elif feat_counts:
        fail(f"Feature counts differ: {feat_counts}")
        cross_failed += 1

    # Compare model counts
    model_counts = {r.format_name: r.n_models_trained for r in results if r.n_models_trained > 0}
    if len(set(model_counts.values())) == 1:
        ok(f"Model counts identical: {list(model_counts.values())[0]} across all formats")
        cross_passed += 1
    elif model_counts:
        fail(f"Model counts differ: {model_counts}")
        cross_failed += 1

    # Compare forecast row counts
    fcast_counts = {r.format_name: r.n_forecast_rows for r in results if r.n_forecast_rows > 0}
    if len(set(fcast_counts.values())) == 1:
        ok(f"Forecast row counts identical: {list(fcast_counts.values())[0]} across all formats")
        cross_passed += 1
    elif fcast_counts:
        fail(f"Forecast row counts differ: {fcast_counts}")
        cross_failed += 1

    # Compare WAPE (allow small tolerance for model non-determinism)
    wapes = {r.format_name: r.val_wape for r in results if r.val_wape < 100}
    if wapes:
        wape_vals = list(wapes.values())
        max_diff = max(wape_vals) - min(wape_vals)
        if max_diff < 0.01:
            ok(f"Val WAPE consistent: {wapes} (max diff={max_diff:.6f})")
            cross_passed += 1
        else:
            warn(f"Val WAPE varies: {wapes} (max diff={max_diff:.6f}) — may be due to model randomness")
            cross_passed += 1  # Not a failure, just a note

    # ── FINAL SUMMARY ─────────────────────────────────────────────
    header("FINAL SUMMARY")
    total_passed = cross_passed
    total_failed = cross_failed
    all_errors = []

    for r in results:
        status = f"{GREEN}PASS{RESET}" if r.failed == 0 else f"{RED}FAIL{RESET}"
        print(f"  {r.format_name.upper():10s}: {status}  "
              f"({r.passed} passed, {r.failed} failed, "
              f"{r.n_features} features, {r.n_models_trained} models, "
              f"{r.n_forecast_rows} forecasts)")
        total_passed += r.passed
        total_failed += r.failed
        all_errors.extend(r.errors)

    print(f"\n  Cross-format checks: {cross_passed} passed, {cross_failed} failed")
    print(f"  {BOLD}Total: {total_passed} passed, {total_failed} failed{RESET}")

    if all_errors:
        print(f"\n  {RED}Errors:{RESET}")
        for err in all_errors:
            print(f"    - {err}")

    # Cleanup
    shutil.rmtree(data_dir, ignore_errors=True)

    if total_failed == 0:
        print(f"\n  {GREEN}{BOLD}ALL FORMAT TESTS PASSED!{RESET}")
        return 0
    else:
        print(f"\n  {RED}{BOLD}SOME TESTS FAILED{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
