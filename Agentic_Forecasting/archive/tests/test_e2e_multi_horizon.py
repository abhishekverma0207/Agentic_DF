"""
End-to-End Simulation Test: Multi-Horizon Pipeline
====================================================

Validates the FULL pipeline from EDA → Segmentation → Feature Engineering
→ Model Training (standard + multi-horizon) → Inference → Backtesting.

Key verifications:
1. Model allocation intelligence is preserved after multi-horizon integration
2. Forward forecasts are generated for ALL horizon weeks (not truncated)
3. build_horizon_weights scales correctly to any max_horizon
4. Multi-horizon models participate correctly in inference
5. The variable-shadowing bug fix in inference.py is verified
6. Evaluation at lag 4-5 has data available
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_test")

# ─── Results collector ───────────────────────────────────────────────────────
_results: List[Dict[str, Any]] = []

def _record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    _results.append({"name": name, "status": status, "detail": detail})
    icon = "OK" if passed else "FAIL"
    logger.info(f"  [{icon}] {name}" + (f" – {detail}" if detail else ""))


def _print_summary():
    print("\n" + "=" * 72)
    print("END-TO-END TEST SUMMARY")
    print("=" * 72)
    n_pass = sum(1 for r in _results if r["status"] == "PASS")
    n_fail = sum(1 for r in _results if r["status"] == "FAIL")
    for r in _results:
        icon = "OK" if r["status"] == "PASS" else "FAIL"
        line = f"  [{icon}] {r['name']}"
        if r["detail"]:
            line += f"  ({r['detail']})"
        print(line)
    print("-" * 72)
    print(f"  Total: {len(_results)}  |  Passed: {n_pass}  |  Failed: {n_fail}")
    print("=" * 72)
    return n_fail == 0


# ═══════════════════════════════════════════════════════════════════════
# PHASE 0 – Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    n_keys: int = 20,
    n_weeks: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic demand data with multiple demand patterns:
      - smooth, erratic, intermittent, lumpy
    """
    rng = np.random.RandomState(seed)
    rows = []

    # Generate year_week values (YYYYWW format)
    base_year = 2022
    base_week = 1
    year_weeks = []
    y, w = base_year, base_week
    for _ in range(n_weeks):
        year_weeks.append(f"{y}{w:02d}")
        w += 1
        if w > 52:
            w = 1
            y += 1

    patterns = ["smooth", "erratic", "intermittent", "lumpy"]

    for ki in range(n_keys):
        key = f"KEY_{ki:04d}"
        pattern = patterns[ki % len(patterns)]

        for idx, yw in enumerate(year_weeks):
            seasonal = 15 * np.sin(2 * np.pi * idx / 52)

            if pattern == "smooth":
                base = 200 + 0.3 * idx + seasonal
                noise = rng.normal(0, 8)
                demand = max(0, base + noise)
            elif pattern == "erratic":
                base = 150 + seasonal
                noise = rng.normal(0, 60)
                demand = max(0, base + noise)
            elif pattern == "intermittent":
                if rng.random() < 0.4:
                    demand = 0.0
                else:
                    demand = max(0, 80 + seasonal + rng.normal(0, 10))
            else:  # lumpy
                if rng.random() < 0.55:
                    demand = 0.0
                else:
                    demand = max(0, rng.exponential(120) + seasonal)

            rows.append({
                "key": key,
                "year_week": yw,
                "demand": round(demand, 2),
                "price": round(rng.uniform(5, 50), 2),
                "promo_flag": int(rng.random() < 0.15),
                "store_count": rng.randint(1, 12),
                "category": f"CAT_{ki % 4}",
            })

    df = pd.DataFrame(rows)
    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 – build_horizon_weights unit tests
# ═══════════════════════════════════════════════════════════════════════

def test_build_horizon_weights():
    logger.info("PHASE 1: Testing build_horizon_weights …")
    from utils.multi_horizon_training import build_horizon_weights

    # --- 1a. Default scheme, H=5 (backward compat) ---
    hw5 = build_horizon_weights(5, scheme="default")
    _record(
        "HW: default H=5 has 5 keys",
        len(hw5) == 5 and set(hw5.keys()) == {1, 2, 3, 4, 5},
    )
    _record(
        "HW: default H=5 sums to 1",
        abs(sum(hw5.values()) - 1.0) < 1e-9,
        f"sum={sum(hw5.values()):.6f}",
    )
    _record(
        "HW: default H=5 monotonically increasing",
        all(hw5[h] <= hw5[h + 1] for h in range(1, 5)),
    )

    # --- 1b. Default scheme, H=10 ---
    hw10 = build_horizon_weights(10, scheme="default")
    _record(
        "HW: default H=10 has 10 keys",
        len(hw10) == 10 and set(hw10.keys()) == set(range(1, 11)),
    )
    _record(
        "HW: default H=10 sums to 1",
        abs(sum(hw10.values()) - 1.0) < 1e-9,
        f"sum={sum(hw10.values()):.6f}",
    )

    # --- 1c. Default scheme, H=13 (UK configs) ---
    hw13 = build_horizon_weights(13, scheme="default")
    _record(
        "HW: default H=13 has 13 keys",
        len(hw13) == 13,
    )
    _record(
        "HW: default H=13 sums to 1",
        abs(sum(hw13.values()) - 1.0) < 1e-9,
    )

    # --- 1d. Balanced scheme ---
    hw_bal = build_horizon_weights(10, scheme="balanced")
    expected_w = 0.1
    _record(
        "HW: balanced H=10 all weights equal",
        all(abs(v - expected_w) < 1e-9 for v in hw_bal.values()),
    )

    # --- 1e. Eval-focused scheme (default eval_lags) ---
    hw_eval = build_horizon_weights(10, scheme="eval_focused")
    _record(
        "HW: eval_focused H=10 sums to 1",
        abs(sum(hw_eval.values()) - 1.0) < 1e-9,
    )
    # Last two lags should have heavy weight (60% / 2 = 0.30 each)
    _record(
        "HW: eval_focused H=10 last two lags heavy",
        hw_eval[9] > hw_eval[1] and hw_eval[10] > hw_eval[1],
        f"lag9={hw_eval[9]:.3f} lag10={hw_eval[10]:.3f} lag1={hw_eval[1]:.3f}",
    )

    # --- 1f. Eval-focused with custom lags ---
    hw_eval_custom = build_horizon_weights(10, scheme="eval_focused", eval_lags=[4, 5])
    _record(
        "HW: eval_focused custom lags [4,5] heavy on 4 and 5",
        hw_eval_custom[4] > hw_eval_custom[1] and hw_eval_custom[5] > hw_eval_custom[1],
        f"lag4={hw_eval_custom[4]:.3f} lag5={hw_eval_custom[5]:.3f}",
    )

    # --- 1g. Edge case H=1 ---
    hw1 = build_horizon_weights(1, scheme="default")
    _record("HW: H=1 returns {1: 1.0}", hw1 == {1: 1.0})

    # --- 1h. Edge case H=0 raises ---
    try:
        build_horizon_weights(0)
        _record("HW: H=0 raises ValueError", False, "no exception raised")
    except ValueError:
        _record("HW: H=0 raises ValueError", True)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 – Model allocation intelligence preservation
# ═══════════════════════════════════════════════════════════════════════

def test_model_allocation_intelligence():
    logger.info("PHASE 2: Testing model allocation intelligence …")
    from utils.model_selection_intelligence import (
        get_dynamic_model_priority,
        ALLOWED_MODEL_FAMILIES,
        MODEL_CHARACTERISTICS,
        detect_discrete_demand,
    )

    # 2a. Multi-horizon models are in ALLOWED_MODEL_FAMILIES
    mh_models = {"multi_horizon_lightgbm", "multi_horizon_xgboost", "multi_horizon_ensemble"}
    _record(
        "MSI: multi-horizon models in ALLOWED_MODEL_FAMILIES",
        mh_models.issubset(set(ALLOWED_MODEL_FAMILIES)),
    )

    # 2b. Multi-horizon models have MODEL_CHARACTERISTICS entries
    _record(
        "MSI: multi-horizon models in MODEL_CHARACTERISTICS",
        all(m in MODEL_CHARACTERISTICS for m in mh_models),
    )

    # 2c. Dynamic priority promotes multi-horizon for long horizons + smooth demand
    rng = np.random.RandomState(99)
    y_smooth = 100 + rng.normal(0, 5, 200)
    y_smooth = np.clip(y_smooth, 0, None)

    priority = get_dynamic_model_priority(
        demand_pattern="smooth",
        cv=0.05,
        zero_fraction=0.01,
        y=y_smooth,
        n_observations=200,
        has_external_features=True,
        target_horizon=10,
        enable_multi_horizon=True,
    )
    # At least one multi-horizon model should be in top 3
    top3 = priority[:3]
    has_mh = any("multi_horizon" in m for m in top3)
    _record(
        "MSI: multi-horizon in top-3 for smooth + horizon=10",
        has_mh,
        f"top3={top3}",
    )

    # 2d. Multi-horizon NOT promoted for very sparse demand
    y_sparse = np.zeros(200)
    y_sparse[::10] = rng.exponential(50, 20)
    priority_sparse = get_dynamic_model_priority(
        demand_pattern="lumpy",
        cv=2.5,
        zero_fraction=0.90,
        y=y_sparse,
        n_observations=200,
        has_external_features=True,
        target_horizon=10,
        enable_multi_horizon=True,
    )
    top3_sparse = priority_sparse[:3]
    has_mh_sparse = any("multi_horizon" in m for m in top3_sparse)
    _record(
        "MSI: multi-horizon NOT in top-3 for very sparse demand",
        not has_mh_sparse,
        f"top3={top3_sparse}",
    )

    # 2e. Pattern-specific model allocation – smooth should favor lgbm/xgb
    priority_smooth = get_dynamic_model_priority(
        demand_pattern="smooth", cv=0.3, zero_fraction=0.02,
        n_observations=200, enable_multi_horizon=False,
    )
    _record(
        "MSI: smooth pattern favors lightgbm or xgboost first",
        priority_smooth[0] in ("lightgbm", "xgboost", "catboost"),
        f"top1={priority_smooth[0]}",
    )

    # 2f. Pattern-specific – intermittent should have zero_inflated / hurdle
    priority_inter = get_dynamic_model_priority(
        demand_pattern="intermittent", cv=0.4, zero_fraction=0.6,
        n_observations=200, enable_multi_horizon=False,
    )
    has_specialist = any(
        m in priority_inter[:5]
        for m in ("zero_inflated", "hurdle_model", "tweedie")
    )
    _record(
        "MSI: intermittent includes specialist models in top-5",
        has_specialist,
        f"top5={priority_inter[:5]}",
    )

    # 2g. Discrete demand detection
    y_discrete = np.array([0, 200, 400, 0, 200, 0, 400, 200, 0, 200] * 20)
    is_disc, n_uniq = detect_discrete_demand(y_discrete)
    _record(
        "MSI: detect_discrete_demand identifies {0,200,400}",
        is_disc and n_uniq <= 5,
        f"is_discrete={is_disc}, n_unique={n_uniq}",
    )


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 – Multi-horizon training (all strategies)
# ═══════════════════════════════════════════════════════════════════════

def test_multi_horizon_training():
    logger.info("PHASE 3: Testing multi-horizon training strategies …")
    from utils.multi_horizon_training import (
        train_multi_horizon_lightgbm,
        train_multi_horizon_xgboost,
        train_multi_horizon_ensemble,
        train_multi_horizon_model,
        MultiHorizonModel,
        MultiHorizonTrainingResult,
        build_horizon_weights,
        create_multi_horizon_targets,
    )

    rng = np.random.RandomState(42)
    n_samples = 300
    n_features = 10
    X = rng.randn(n_samples, n_features)
    y = 100 + 5 * X[:, 0] - 3 * X[:, 1] + rng.normal(0, 2, n_samples)
    y = np.clip(y, 0, None)

    # Split 70/30
    split = int(0.7 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    max_horizon = 10  # Test with config-like horizon

    # 3a. Direct-separate LightGBM
    result_lgb = train_multi_horizon_lightgbm(
        X_train, y_train, X_val, y_val,
        max_horizon=max_horizon, target_horizon=max_horizon,
        strategy="direct_separate",
    )
    _record(
        "MH-Train: direct_separate LightGBM returns result",
        isinstance(result_lgb, MultiHorizonTrainingResult),
    )
    _record(
        "MH-Train: LGB result has correct max_horizon",
        result_lgb.max_horizon == max_horizon,
    )
    _record(
        "MH-Train: LGB has wape for all horizons",
        len(result_lgb.val_wape_by_horizon) == max_horizon,
        f"horizons={sorted(result_lgb.val_wape_by_horizon.keys())}",
    )
    _record(
        "MH-Train: LGB horizon_weights has correct count",
        len(result_lgb.horizon_weights) == max_horizon,
    )
    _record(
        "MH-Train: LGB horizon_weights sum to 1",
        abs(sum(result_lgb.horizon_weights.values()) - 1.0) < 1e-9,
    )

    # 3b. Predict with model for all horizons
    model_lgb = result_lgb.model
    preds_all = model_lgb.predict(X_val[:5], return_all_horizons=True)
    _record(
        "MH-Predict: return_all_horizons gives dict with max_horizon keys",
        isinstance(preds_all, dict) and len(preds_all) == max_horizon,
        f"horizons={sorted(preds_all.keys())}",
    )
    # Each horizon should produce same number of predictions
    for h, p in preds_all.items():
        _record(
            f"MH-Predict: horizon {h} produces 5 predictions",
            len(p) == 5,
        )
    # All predictions non-negative
    all_nonneg = all(np.all(p >= 0) for p in preds_all.values())
    _record("MH-Predict: all predictions non-negative", all_nonneg)

    # 3c. Direct-separate XGBoost
    result_xgb = train_multi_horizon_xgboost(
        X_train, y_train, X_val, y_val,
        max_horizon=max_horizon, target_horizon=max_horizon,
        strategy="direct_separate",
    )
    _record(
        "MH-Train: direct_separate XGBoost returns result",
        isinstance(result_xgb, MultiHorizonTrainingResult),
    )
    _record(
        "MH-Train: XGB has wape for all horizons",
        len(result_xgb.val_wape_by_horizon) == max_horizon,
    )

    # 3d. Horizon-weighted single model
    result_hw = train_multi_horizon_lightgbm(
        X_train, y_train, X_val, y_val,
        max_horizon=max_horizon, target_horizon=max_horizon,
        strategy="horizon_weighted",
    )
    _record(
        "MH-Train: horizon_weighted returns result",
        isinstance(result_hw, MultiHorizonTrainingResult),
    )
    _record(
        "MH-Train: horizon_weighted has all horizon wapes",
        len(result_hw.val_wape_by_horizon) == max_horizon,
    )

    # 3e. Target-only
    result_tgt = train_multi_horizon_lightgbm(
        X_train, y_train, X_val, y_val,
        max_horizon=max_horizon, target_horizon=max_horizon,
        strategy="target_only",
    )
    _record(
        "MH-Train: target_only returns result",
        isinstance(result_tgt, MultiHorizonTrainingResult),
    )

    # 3f. Ensemble (LightGBM + XGBoost)
    result_ens = train_multi_horizon_ensemble(
        X_train, y_train, X_val, y_val,
        max_horizon=max_horizon, target_horizon=max_horizon,
    )
    _record(
        "MH-Train: ensemble returns result",
        isinstance(result_ens, MultiHorizonTrainingResult),
    )
    _record(
        "MH-Train: ensemble strategy is 'ensemble'",
        result_ens.strategy == "ensemble",
    )

    # 3g. Generic entry point
    result_gen = train_multi_horizon_model(
        X_train, y_train, X_val, y_val,
        model_type="lightgbm", max_horizon=max_horizon, target_horizon=max_horizon,
    )
    _record(
        "MH-Train: generic entry point works",
        isinstance(result_gen, MultiHorizonTrainingResult),
    )

    # 3h. create_multi_horizon_targets correctness
    y_test = np.arange(20, dtype=float)
    targets = create_multi_horizon_targets(y_test, max_horizon=5)
    _record(
        "MH-Data: targets shape correct",
        targets.shape == (15, 5),
        f"shape={targets.shape}",
    )
    # targets[0] should be [y[1], y[2], y[3], y[4], y[5]]
    expected_first_row = np.array([1, 2, 3, 4, 5], dtype=float)
    _record(
        "MH-Data: targets[0] == [1,2,3,4,5]",
        np.allclose(targets[0], expected_first_row),
        f"targets[0]={targets[0]}",
    )


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 – Training registry integration
# ═══════════════════════════════════════════════════════════════════════

def test_training_registry():
    logger.info("PHASE 4: Testing training registry integration …")
    from utils.model_training import (
        TRAINING_REGISTRY,
        FEATURE_BASED_MODELS,
        is_feature_based,
        train_model_by_name,
        MULTI_HORIZON_AVAILABLE,
    )

    # 4a. Multi-horizon available
    _record("Registry: MULTI_HORIZON_AVAILABLE", MULTI_HORIZON_AVAILABLE)

    # 4b. Multi-horizon models in registry
    mh_in_registry = all(
        m in TRAINING_REGISTRY
        for m in ("multi_horizon_lightgbm", "multi_horizon_xgboost", "multi_horizon_ensemble")
    )
    _record("Registry: multi-horizon models registered", mh_in_registry)

    # 4c. Multi-horizon models are feature-based
    mh_feature = all(
        m in FEATURE_BASED_MODELS
        for m in ("multi_horizon_lightgbm", "multi_horizon_xgboost", "multi_horizon_ensemble")
    )
    _record("Registry: multi-horizon models are feature-based", mh_feature)

    # 4d. train_model_by_name works with multi-horizon + kwargs
    rng = np.random.RandomState(7)
    n = 200
    X = rng.randn(n, 5)
    y = np.clip(50 + 3 * X[:, 0] + rng.normal(0, 2, n), 0, None)
    s = int(0.7 * n)

    result = train_model_by_name(
        "multi_horizon_lightgbm",
        X[:s], y[:s], X[s:], y[s:],
        max_horizon=8, target_horizon=8,
    )
    _record(
        "Registry: train_model_by_name(multi_horizon_lightgbm) works",
        result is not None and hasattr(result, "model"),
    )
    # Verify max_horizon was respected
    model = result.model
    if hasattr(model, "max_horizon"):
        _record(
            "Registry: model has correct max_horizon=8",
            model.max_horizon == 8,
            f"max_horizon={model.max_horizon}",
        )
    else:
        _record("Registry: model has correct max_horizon=8", False, "no max_horizon attr")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5 – Inference variable shadowing fix verification
# ═══════════════════════════════════════════════════════════════════════

def test_inference_variable_shadowing_fix():
    """Verify the forecast_horizon variable is not overwritten inside the loop."""
    logger.info("PHASE 5: Testing inference variable shadowing fix …")
    import ast

    inference_path = PROJECT_ROOT / "utils" / "inference.py"
    source = inference_path.read_text()

    # Parse AST and find the generate_forward_forecasts function
    tree = ast.parse(source)

    found_shadow = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_forward_forecasts":
            # Walk through the function body looking for assignments
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == "forecast_horizon":
                            # Check if it's inside a for loop (the shadowing bug)
                            # The legitimate assignment is at function start level
                            # The bug was: forecast_horizon = step_idx + 1
                            if isinstance(child.value, ast.BinOp):
                                # This is the bug pattern: forecast_horizon = step_idx + 1
                                found_shadow = True

    _record(
        "Shadow-fix: no forecast_horizon reassignment in loop",
        not found_shadow,
        "forecast_horizon is no longer overwritten by step_idx+1" if not found_shadow
        else "BUG STILL PRESENT",
    )

    # Also verify current_horizon_step is used instead
    has_current_horizon = "current_horizon_step" in source
    _record(
        "Shadow-fix: current_horizon_step variable exists",
        has_current_horizon,
    )


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6 – Forward forecast generation (simulated)
# ═══════════════════════════════════════════════════════════════════════

def test_forward_forecast_all_horizons():
    """
    Simulate the forward forecast loop to verify ALL horizon steps are generated.
    This mimics generate_forward_forecasts without needing the full pipeline.
    """
    logger.info("PHASE 6: Testing forward forecast generation for all horizons …")
    from utils.multi_horizon_training import train_multi_horizon_lightgbm, MultiHorizonModel

    # Train a simple model
    rng = np.random.RandomState(42)
    n = 300
    n_feat = 8
    X = rng.randn(n, n_feat)
    y = np.clip(80 + 4 * X[:, 0] - 2 * X[:, 1] + rng.normal(0, 3, n), 0, None)
    s = int(0.7 * n)

    forecast_horizon = 10

    result = train_multi_horizon_lightgbm(
        X[:s], y[:s], X[s:], y[s:],
        max_horizon=forecast_horizon, target_horizon=forecast_horizon,
        strategy="direct_separate",
    )
    model = result.model

    # Simulate the inference loop (mirrors generate_forward_forecasts)
    test_features = X[s : s + forecast_horizon]  # 10 rows of features
    historical_actuals = y[:s].tolist()

    all_forecasts = []
    for step_idx in range(forecast_horizon):
        if step_idx >= len(test_features) or step_idx >= forecast_horizon:
            break

        features = test_features[step_idx].copy()
        X_pred = features.reshape(1, -1)

        # This is the FIXED code: uses current_horizon_step, not forecast_horizon
        current_horizon_step = step_idx + 1
        pred = model.predict(X_pred, horizon=current_horizon_step)
        if isinstance(pred, np.ndarray):
            pred = float(pred[0])
        pred = max(0.0, pred)

        historical_actuals.append(pred)
        all_forecasts.append({
            "forecast_step": step_idx + 1,
            "lag": step_idx + 1,
            "predicted": pred,
        })

    # Verify ALL horizon steps were generated
    _record(
        f"Forward: generated {len(all_forecasts)} forecasts (expected {forecast_horizon})",
        len(all_forecasts) == forecast_horizon,
    )

    # Verify lag values are 1 through forecast_horizon
    lags = [f["lag"] for f in all_forecasts]
    expected_lags = list(range(1, forecast_horizon + 1))
    _record(
        f"Forward: lags are [1..{forecast_horizon}]",
        lags == expected_lags,
        f"lags={lags}",
    )

    # Verify evaluation lags 4 and 5 have data
    lag4 = [f for f in all_forecasts if f["lag"] == 4]
    lag5 = [f for f in all_forecasts if f["lag"] == 5]
    _record("Forward: lag 4 has forecast data", len(lag4) == 1)
    _record("Forward: lag 5 has forecast data", len(lag5) == 1)

    # All predictions should be non-negative
    all_nonneg = all(f["predicted"] >= 0 for f in all_forecasts)
    _record("Forward: all predictions non-negative", all_nonneg)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 7 – EDA pipeline
# ═══════════════════════════════════════════════════════════════════════

def test_eda_pipeline(df: pd.DataFrame, output_dir: str):
    logger.info("PHASE 7: Testing EDA pipeline …")
    from utils.eda import run_eda_pipeline

    eda_dir = os.path.join(output_dir, "eda_output")
    os.makedirs(eda_dir, exist_ok=True)

    try:
        eda_result = run_eda_pipeline(
            df=df,
            key_columns=["key"],
            date_col="year_week",
            target_col="demand",
            numeric_features=["price", "promo_flag", "store_count"],
            categorical_features=["category"],
            output_dir=eda_dir,
            period=52,
            verbose=False,
            exhaustive=False,
        )
        _record("EDA: run_eda_pipeline completed", True)

        # Check for per_key_metrics
        pkm_path = os.path.join(eda_dir, "per_key_metrics.csv")
        if os.path.exists(pkm_path):
            pkm = pd.read_csv(pkm_path)
            _record("EDA: per_key_metrics.csv created", True, f"{len(pkm)} keys")
            # Verify demand pattern classification
            if "demand_pattern" in pkm.columns:
                patterns = pkm["demand_pattern"].unique()
                _record(
                    "EDA: demand patterns classified",
                    len(patterns) >= 1,
                    f"patterns={list(patterns)}",
                )
            else:
                _record("EDA: demand_pattern column exists", False)
        else:
            _record("EDA: per_key_metrics.csv created", False)
            pkm = None

        return eda_result, eda_dir
    except Exception as e:
        _record("EDA: run_eda_pipeline completed", False, str(e))
        logger.error(traceback.format_exc())
        return None, eda_dir


# ═══════════════════════════════════════════════════════════════════════
# PHASE 8 – Segmentation pipeline
# ═══════════════════════════════════════════════════════════════════════

def test_segmentation_pipeline(eda_dir: str, output_dir: str):
    logger.info("PHASE 8: Testing segmentation pipeline …")
    from utils.segmentation import run_segmentation_pipeline

    seg_dir = os.path.join(output_dir, "segmentation_output")
    os.makedirs(seg_dir, exist_ok=True)

    pkm_path = os.path.join(eda_dir, "per_key_metrics.csv")
    if not os.path.exists(pkm_path):
        _record("Segmentation: skipped (no per_key_metrics)", False, "depends on EDA")
        return None, seg_dir

    pkm = pd.read_csv(pkm_path)

    try:
        seg_result = run_segmentation_pipeline(
            per_key_metrics=pkm,
            key_cols=["key"],
            output_dir=seg_dir,
            allowed_model_families=[
                "lightgbm", "xgboost", "catboost", "random_forest",
                "zero_inflated", "hurdle_model", "tweedie",
                "multi_horizon_lightgbm", "multi_horizon_xgboost",
            ],
            enable_deep_models=False,
            clustering_method="gmm",
            n_clusters_range=[2, 3, 4],
            create_visualizations=False,
            eda_dir=eda_dir,
        )
        _record("Segmentation: run_segmentation_pipeline completed", True)

        # Check segment count
        _record(
            "Segmentation: produces 2+ segments",
            seg_result.n_segments >= 2,
            f"n_segments={seg_result.n_segments}",
        )

        # Check model recommendations exist
        has_recs = seg_result.model_recommendations is not None
        _record("Segmentation: model_recommendations populated", has_recs)

        if has_recs and isinstance(seg_result.model_recommendations, dict):
            strats = seg_result.model_recommendations.get("segment_strategies", {})
            if strats:
                first_seg = list(strats.values())[0]
                has_pattern = "dominant_pattern" in first_seg
                # Field is 'recommended_models' or 'primary_model' (not 'primary_models')
                has_models = ("recommended_models" in first_seg
                              or "primary_models" in first_seg
                              or "primary_model" in first_seg)
                model_field = (first_seg.get("recommended_models")
                               or first_seg.get("primary_models")
                               or first_seg.get("primary_model"))
                _record(
                    "Segmentation: recommendations have pattern + models",
                    has_pattern and has_models,
                    f"pattern={first_seg.get('dominant_pattern')}, models={model_field}",
                )

        return seg_result, seg_dir
    except Exception as e:
        _record("Segmentation: run_segmentation_pipeline completed", False, str(e))
        logger.error(traceback.format_exc())
        return None, seg_dir


# ═══════════════════════════════════════════════════════════════════════
# PHASE 9 – Feature engineering pipeline
# ═══════════════════════════════════════════════════════════════════════

def test_feature_engineering(df: pd.DataFrame, output_dir: str):
    logger.info("PHASE 9: Testing feature engineering pipeline …")
    from utils.feature_engineering import run_leakage_free_feature_pipeline

    feat_dir = os.path.join(output_dir, "feature_output")
    os.makedirs(feat_dir, exist_ok=True)

    # Get unique weeks sorted
    weeks = sorted(df["year_week"].unique())
    n_w = len(weeks)
    # Split: ~70% train, ~15% val, ~15% test
    train_end_idx = int(n_w * 0.70)
    val_end_idx = int(n_w * 0.85)

    train_start = weeks[0]
    train_end = weeks[train_end_idx - 1]
    val_start = weeks[train_end_idx]
    val_end = weeks[val_end_idx - 1]
    test_start = weeks[val_end_idx]
    test_end = weeks[-1]

    logger.info(f"  Splits: train={train_start}-{train_end}, val={val_start}-{val_end}, test={test_start}-{test_end}")

    try:
        fe_result = run_leakage_free_feature_pipeline(
            df=df,
            key_cols=["key"],
            date_col="year_week",
            target_col="demand",
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
            forecast_lag=1,
            categorical_cols=["category"],
            output_dir=feat_dir,
            target_lags=[1, 2, 4, 8],
            rolling_windows=[4, 8],
            rolling_stats=["mean", "std"],
            include_intermittency_features=True,
            include_trend_features=True,
            include_fourier_features=True,
            fourier_order=2,
            seasonal_period=52,
        )
        _record("FE: run_leakage_free_feature_pipeline completed", True)

        # Check output files
        for split_name in ["train_features.csv", "val_features.csv", "test_features.csv"]:
            path = os.path.join(feat_dir, split_name)
            exists = os.path.exists(path)
            if exists:
                split_df = pd.read_csv(path)
                _record(f"FE: {split_name} created", True, f"{len(split_df)} rows, {len(split_df.columns)} cols")
            else:
                _record(f"FE: {split_name} created", exists)

        # Check feature count
        _record(
            "FE: features created",
            fe_result.n_features_created > 5,
            f"n_features={fe_result.n_features_created}",
        )

        return fe_result, feat_dir, {
            "train_start": train_start, "train_end": train_end,
            "val_start": val_start, "val_end": val_end,
            "test_start": test_start, "test_end": test_end,
        }
    except Exception as e:
        _record("FE: run_leakage_free_feature_pipeline completed", False, str(e))
        logger.error(traceback.format_exc())
        return None, feat_dir, None


# ═══════════════════════════════════════════════════════════════════════
# PHASE 10 – Model training with multi-horizon on real features
# ═══════════════════════════════════════════════════════════════════════

def test_model_training_with_features(feat_dir: str, forecast_horizon: int = 10):
    logger.info("PHASE 10: Testing model training on engineered features …")
    from utils.model_training import train_model_by_name, train_best_model_for_segment
    from utils.multi_horizon_training import train_multi_horizon_lightgbm

    train_path = os.path.join(feat_dir, "train_features.csv")
    val_path = os.path.join(feat_dir, "val_features.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        _record("Training: skipped (no feature files)", False, "depends on FE")
        return None

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Get feature columns (exclude meta columns)
    meta_cols = {"key", "year_week", "demand", "split", "segment_id",
                 "demand_pattern", "model_level", "model_group", "label",
                 "demand_log", "category"}
    feature_cols = [c for c in train_df.columns if c not in meta_cols and not c.startswith("__")]

    # Filter numeric only
    feature_cols = [c for c in feature_cols if train_df[c].dtype in ("float64", "int64", "float32", "int32")]

    if not feature_cols:
        _record("Training: skipped (no numeric features)", False)
        return None

    logger.info(f"  Using {len(feature_cols)} features for training")

    target_col = "demand"
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = train_df[target_col].fillna(0).values
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_val = val_df[target_col].fillna(0).values

    trained_models = {}

    # 10a. Standard LightGBM
    try:
        result_lgb = train_model_by_name("lightgbm", X_train, y_train, X_val, y_val)
        _record(
            "Training: standard lightgbm trained",
            True,
            f"val_wape={result_lgb.val_wape:.4f}",
        )
        trained_models["lightgbm"] = result_lgb
    except Exception as e:
        _record("Training: standard lightgbm trained", False, str(e))

    # 10b. Multi-horizon LightGBM
    try:
        result_mh = train_multi_horizon_lightgbm(
            X_train, y_train, X_val, y_val,
            max_horizon=forecast_horizon, target_horizon=forecast_horizon,
            strategy="direct_separate",
        )
        _record(
            "Training: multi-horizon lightgbm trained",
            True,
            f"val_wape@H{forecast_horizon}={result_mh.val_target_wape:.4f}",
        )
        trained_models["multi_horizon_lightgbm"] = result_mh

        # Verify it can predict at each horizon
        for h in range(1, min(forecast_horizon + 1, 4)):
            pred = result_mh.model.predict(X_val[:3], horizon=h)
            _record(
                f"Training: MH model predicts at horizon {h}",
                len(pred) == 3 and np.all(pred >= 0),
            )
    except Exception as e:
        _record("Training: multi-horizon lightgbm trained", False, str(e))
        logger.error(traceback.format_exc())

    # 10c. XGBoost
    try:
        result_xgb = train_model_by_name("xgboost", X_train, y_train, X_val, y_val)
        _record(
            "Training: standard xgboost trained",
            True,
            f"val_wape={result_xgb.val_wape:.4f}",
        )
        trained_models["xgboost"] = result_xgb
    except Exception as e:
        _record("Training: standard xgboost trained", False, str(e))

    # 10d. Multi-horizon via train_model_by_name
    try:
        result_mh2 = train_model_by_name(
            "multi_horizon_lightgbm",
            X_train, y_train, X_val, y_val,
            max_horizon=forecast_horizon, target_horizon=forecast_horizon,
        )
        _record(
            "Training: train_model_by_name(multi_horizon_lgb) with kwargs",
            True,
        )
    except Exception as e:
        _record("Training: train_model_by_name(multi_horizon_lgb) with kwargs", False, str(e))

    return trained_models


# ═══════════════════════════════════════════════════════════════════════
# PHASE 11 – Simulated inference across all horizon steps
# ═══════════════════════════════════════════════════════════════════════

def test_simulated_inference(feat_dir: str, trained_models: dict, forecast_horizon: int = 10):
    logger.info("PHASE 11: Simulated inference across all horizons …")

    test_path = os.path.join(feat_dir, "test_features.csv")
    if not os.path.exists(test_path):
        _record("Inference-sim: skipped (no test features)", False)
        return

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(os.path.join(feat_dir, "train_features.csv"))

    meta_cols = {"key", "year_week", "demand", "split", "segment_id",
                 "demand_pattern", "model_level", "model_group", "label",
                 "demand_log", "category"}
    feature_cols = [c for c in train_df.columns
                    if c not in meta_cols
                    and not c.startswith("__")
                    and train_df[c].dtype in ("float64", "int64", "float32", "int32")]

    # Test with both standard and multi-horizon models
    for model_name, model_result in trained_models.items():
        model = model_result.model if hasattr(model_result, "model") else model_result
        is_mh = "multi_horizon" in model_name

        # Pick first key
        keys = test_df["key"].unique()
        if len(keys) == 0:
            continue
        key = keys[0]
        key_data = test_df[test_df["key"] == key].sort_values("year_week")

        avail_cols = [c for c in feature_cols if c in key_data.columns]
        features_matrix = key_data[avail_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

        # Simulate recursive forecast loop
        forecasts = []
        max_steps = min(forecast_horizon, len(features_matrix))
        for step_idx in range(max_steps):
            X = features_matrix[step_idx].reshape(1, -1)
            current_horizon_step = step_idx + 1

            if is_mh:
                pred = model.predict(X, horizon=current_horizon_step)
            else:
                pred = model.predict(X)

            if isinstance(pred, np.ndarray):
                pred = float(pred[0])
            pred = max(0.0, pred)
            forecasts.append({"lag": current_horizon_step, "predicted": pred})

        _record(
            f"Inference-sim ({model_name}): {len(forecasts)} steps generated",
            len(forecasts) == max_steps,
        )

        # Verify lag coverage
        lags = [f["lag"] for f in forecasts]
        expected = list(range(1, max_steps + 1))
        _record(
            f"Inference-sim ({model_name}): lags=[1..{max_steps}]",
            lags == expected,
        )

        # Verify lag 4-5 available
        if max_steps >= 5:
            lag4 = [f for f in forecasts if f["lag"] == 4]
            lag5 = [f for f in forecasts if f["lag"] == 5]
            _record(
                f"Inference-sim ({model_name}): lag4+lag5 have forecasts",
                len(lag4) == 1 and len(lag5) == 1,
            )


# ═══════════════════════════════════════════════════════════════════════
# PHASE 12 – Simulated backtesting with multi-horizon models
# ═══════════════════════════════════════════════════════════════════════

def test_simulated_backtest(feat_dir: str, trained_models: dict, forecast_horizon: int = 10):
    logger.info("PHASE 12: Simulated backtesting with multi-horizon …")

    train_path = os.path.join(feat_dir, "train_features.csv")
    val_path = os.path.join(feat_dir, "val_features.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        _record("Backtest-sim: skipped (no feature files)", False)
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    meta_cols = {"key", "year_week", "demand", "split", "segment_id",
                 "demand_pattern", "model_level", "model_group", "label",
                 "demand_log", "category"}
    feature_cols = [c for c in train_df.columns
                    if c not in meta_cols
                    and not c.startswith("__")
                    and train_df[c].dtype in ("float64", "int64", "float32", "int32")]

    from utils.multi_horizon_training import train_multi_horizon_lightgbm, compute_wape

    # Simulate 3 rolling origins
    keys = train_df["key"].unique()
    combined = pd.concat([train_df, val_df], ignore_index=True)
    all_weeks = sorted(combined["year_week"].unique())
    n_weeks_total = len(all_weeks)

    n_origins = 3
    origin_results = []

    for origin_idx in range(n_origins):
        # Shift cutoff forward — ensure enough room for forecast_horizon after cutoff
        cutoff_idx = int(n_weeks_total * 0.50) + origin_idx * 2
        if cutoff_idx >= n_weeks_total - forecast_horizon:
            break

        train_cutoff = all_weeks[cutoff_idx]

        origin_train = combined[combined["year_week"] <= train_cutoff]
        origin_test = combined[combined["year_week"] > train_cutoff]

        if len(origin_test) == 0:
            continue

        # Prepare data
        avail_cols = [c for c in feature_cols if c in origin_train.columns]
        X_tr = origin_train[avail_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_tr = origin_train["demand"].fillna(0).values
        X_te = origin_test[avail_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_te = origin_test["demand"].fillna(0).values

        if len(X_tr) < 50 or len(X_te) < 5:
            continue

        # For multi-horizon training, max_horizon must be < min rows per series.
        # Use a safe horizon based on per-key test rows.
        test_weeks_count = origin_test.groupby("key")["year_week"].nunique().min()
        mh = min(forecast_horizon, max(int(test_weeks_count) - 1, 2))

        # Retrain at each origin (use pooled data, train on rows, validate on a subset)
        # For simplicity, use first 80% of train data for training, last 20% for validation
        n_tr = len(X_tr)
        tr_split = int(n_tr * 0.8)
        try:
            result = train_multi_horizon_lightgbm(
                X_tr[:tr_split], y_tr[:tr_split],
                X_tr[tr_split:], y_tr[tr_split:],
                max_horizon=mh, target_horizon=mh,
                strategy="direct_separate",
            )

            # Generate forecasts per key
            origin_forecasts = []
            for key in keys[:5]:  # Subset for speed
                key_test = origin_test[origin_test["key"] == key].sort_values("year_week")
                if len(key_test) == 0:
                    continue
                key_features = key_test[avail_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
                key_actuals = key_test["demand"].fillna(0).values

                for step_idx in range(min(mh, len(key_features))):
                    X = key_features[step_idx].reshape(1, -1)
                    current_horizon_step = step_idx + 1
                    pred = result.model.predict(X, horizon=current_horizon_step)
                    if isinstance(pred, np.ndarray):
                        pred = float(pred[0])
                    pred = max(0.0, pred)
                    actual = float(key_actuals[step_idx]) if step_idx < len(key_actuals) else 0.0
                    origin_forecasts.append({
                        "origin": origin_idx,
                        "key": key,
                        "lag": current_horizon_step,
                        "predicted": pred,
                        "actual": actual,
                    })

            origin_results.extend(origin_forecasts)
        except Exception as e:
            logger.warning(f"Origin {origin_idx} failed: {e}")
            traceback.print_exc()
            continue

    if origin_results:
        bt_df = pd.DataFrame(origin_results)
        total_origins = bt_df["origin"].nunique()
        total_forecasts = len(bt_df)

        _record(
            f"Backtest-sim: {total_origins} origins, {total_forecasts} forecasts",
            total_origins >= 1 and total_forecasts >= 10,
        )

        # Check lag distribution
        lag_counts = bt_df["lag"].value_counts().sort_index()
        max_lag = lag_counts.index.max()
        _record(
            f"Backtest-sim: max lag = {max_lag}",
            max_lag >= 4,
        )

        # Compute WAPE at lag 4-5 if available
        for eval_lag in [4, 5]:
            lag_data = bt_df[bt_df["lag"] == eval_lag]
            if len(lag_data) > 0:
                total_actual = lag_data["actual"].sum()
                if total_actual > 0:
                    wape = np.sum(np.abs(lag_data["actual"] - lag_data["predicted"])) / total_actual
                    _record(
                        f"Backtest-sim: WAPE @ lag {eval_lag}",
                        True,
                        f"WAPE={wape:.4f} ({len(lag_data)} records)",
                    )
                else:
                    _record(
                        f"Backtest-sim: WAPE @ lag {eval_lag}",
                        True,
                        f"all actuals zero ({len(lag_data)} records)",
                    )
            else:
                _record(f"Backtest-sim: WAPE @ lag {eval_lag}", False, "no data at this lag")
    else:
        _record("Backtest-sim: generated forecasts", False, "no forecasts produced")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("HARMONIQ DEMAND-IQ: End-to-End Multi-Horizon Simulation Test")
    print("=" * 72)

    # Create temp directory for test artifacts
    test_dir = tempfile.mkdtemp(prefix="harmoniq_e2e_test_")
    logger.info(f"Test artifacts: {test_dir}")

    forecast_horizon = 10

    try:
        # Generate synthetic data
        logger.info("Generating synthetic demand data …")
        df = generate_synthetic_data(n_keys=20, n_weeks=100, seed=42)
        data_path = os.path.join(test_dir, "synthetic_data.csv")
        df.to_csv(data_path, index=False)
        logger.info(f"  Created {len(df)} rows, {df['key'].nunique()} keys, {df['year_week'].nunique()} weeks")
        _record("Data: synthetic data generated", True, f"{len(df)} rows")

        # PHASE 1: build_horizon_weights
        test_build_horizon_weights()

        # PHASE 2: Model allocation intelligence
        test_model_allocation_intelligence()

        # PHASE 3: Multi-horizon training
        test_multi_horizon_training()

        # PHASE 4: Training registry
        test_training_registry()

        # PHASE 5: Variable shadowing fix
        test_inference_variable_shadowing_fix()

        # PHASE 6: Forward forecast all horizons
        test_forward_forecast_all_horizons()

        # PHASE 7: EDA pipeline
        eda_result, eda_dir = test_eda_pipeline(df, test_dir)

        # PHASE 8: Segmentation pipeline
        seg_result, seg_dir = test_segmentation_pipeline(eda_dir, test_dir)

        # PHASE 9: Feature engineering
        fe_result, feat_dir, splits = test_feature_engineering(df, test_dir)

        # PHASE 10: Model training with features
        trained_models = test_model_training_with_features(feat_dir, forecast_horizon)

        # PHASE 11: Simulated inference
        if trained_models:
            test_simulated_inference(feat_dir, trained_models, forecast_horizon)
        else:
            _record("Inference-sim: skipped (no trained models)", False)

        # PHASE 12: Simulated backtesting
        if trained_models:
            test_simulated_backtest(feat_dir, trained_models, forecast_horizon)
        else:
            _record("Backtest-sim: skipped (no trained models)", False)

    finally:
        # Print summary
        all_passed = _print_summary()

        # Cleanup
        try:
            shutil.rmtree(test_dir)
            logger.info(f"Cleaned up test artifacts: {test_dir}")
        except Exception:
            pass

        if not all_passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
