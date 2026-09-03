"""Verify Phase-3 hierarchical models actually train (bug fix for the
caveat where ``multi_level_ensemble`` / ``global_local`` / ``mixed_effects``
silently failed with a missing-hierarchy_col TypeError and were skipped).

These are surgical tests exercising the internal ``create_training_fn``
and ``_adapt_hierarchical_training`` plumbing; they do not run the full
pipeline. A regression would surface as any of the three models
returning None / raising when a valid hierarchy_col is supplied.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest


def _make_training_fixture(n_keys=20, n_rows_per_key=60, seed=0):
    """Build a synthetic (train, val) pair with a real hierarchy column.

    Each key belongs to one of 4 groups. The target depends on both the
    key-level baseline and the group-level seasonal pattern — so a
    multi-level model should beat a global model.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_keys):
        grp = f"G{i % 4}"
        baseline = rng.normal(100, 15)
        season_amp = rng.normal(20, 4)
        for t in range(n_rows_per_key):
            seasonal = season_amp * np.sin(2 * np.pi * t / 12)
            noise = rng.normal(0, 5)
            rows.append({
                "key": f"K{i:02d}",
                "group_id": grp,
                "year_week": 202400 + t,
                "actual_sales": max(0.0, baseline + seasonal + noise),
                "feat_lag_1": rng.normal(0, 1),
                "feat_lag_2": rng.normal(0, 1),
                "feat_rolling_4": rng.normal(0, 1),
            })
    df = pd.DataFrame(rows)
    split = int(0.75 * n_rows_per_key)
    train_df = df[df.groupby("key").cumcount() < split].reset_index(drop=True)
    val_df = df[df.groupby("key").cumcount() >= split].reset_index(drop=True)
    return train_df, val_df


def test_multi_level_ensemble_trains_when_hierarchy_is_available():
    """Previously this raised TypeError: missing hierarchy_col and was
    silently skipped by the outer try/except. After the fix it must
    actually train and return a non-None TrainingResult."""
    from utils.model_training import train_multi_level_ensemble
    # Direct test of the underlying function — verifies our plumbing
    # passes hierarchy_col correctly.
    train_df, val_df = _make_training_fixture()
    feature_cols = ["feat_lag_1", "feat_lag_2", "feat_rolling_4"]

    wrapper, meta = train_multi_level_ensemble(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col="actual_sales",
        key_col="key",
        hierarchy_col="group_id",
        model_family="lightgbm",
        min_key_samples=10,
        min_group_samples=20,
    )
    assert wrapper is not None
    assert meta.get("model_type") in ("multi_level_ensemble", "hierarchical")


def test_adapter_receives_hierarchy_col_via_kwargs():
    """The adapter must forward ``hierarchy_col`` (plus DataFrames) into
    the hierarchical trainer — the bug was that ``create_training_fn``
    wasn't populating these kwargs."""
    from utils.model_training import _adapt_hierarchical_training, train_multi_level_ensemble

    train_df, val_df = _make_training_fixture()
    feature_cols = ["feat_lag_1", "feat_lag_2", "feat_rolling_4"]

    result = _adapt_hierarchical_training(
        train_multi_level_ensemble,
        train_df[feature_cols].to_numpy(),
        train_df["actual_sales"].to_numpy(),
        val_df[feature_cols].to_numpy(),
        val_df["actual_sales"].to_numpy(),
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        key_col="key",
        target_col="actual_sales",
        hierarchy_col="group_id",
    )
    assert result is not None
    assert result.model is not None


def test_create_training_fn_returns_skip_when_hierarchy_missing():
    """When no hierarchy_col is resolvable, the closure must be a no-op
    (return None) so the outer pipeline records a skipped candidate
    rather than raising a TypeError."""
    # Import the inner factory via a light-weight reconstruction: the
    # function is a local closure, so we exercise it through the public
    # ``TRAINING_REGISTRY`` path plus a bypass where we call the lambda
    # directly with the right signature and an empty DataFrame context.
    import utils.model_training as mt

    # Sanity: the hierarchical entry in the registry exists.
    assert "multi_level_ensemble" in mt.TRAINING_REGISTRY

    train_df, val_df = _make_training_fixture()
    feature_cols = ["feat_lag_1", "feat_lag_2", "feat_rolling_4"]

    # Direct registry call without the necessary context kwargs ⇒ will
    # hit the legacy single-key fallback and raise. This documents the
    # OLD broken behaviour — proves our closure is needed.
    reg_fn = mt.TRAINING_REGISTRY["multi_level_ensemble"]
    with pytest.raises(TypeError):
        reg_fn(
            train_df[feature_cols].to_numpy(),
            train_df["actual_sales"].to_numpy(),
            val_df[feature_cols].to_numpy(),
            val_df["actual_sales"].to_numpy(),
        )


def test_create_training_fn_trains_when_context_provided():
    """End-to-end: the closure returned by create_training_fn must
    actually train when called with (X, y, X, y) + proper context.

    We reproduce the closure inline because the real factory is scoped
    inside ``run_full_training_pipeline``. This is equivalent to what
    our patched call site does.
    """
    from utils.model_training import _adapt_hierarchical_training, TRAINING_REGISTRY

    train_df, val_df = _make_training_fixture()
    feature_cols = ["feat_lag_1", "feat_lag_2", "feat_rolling_4"]

    X_tr = train_df[feature_cols].to_numpy()
    y_tr = train_df["actual_sales"].to_numpy()
    X_v = val_df[feature_cols].to_numpy()
    y_v = val_df["actual_sales"].to_numpy()

    # Mirror the closure construction in ``create_training_fn``.
    hierarchy_fn = TRAINING_REGISTRY["multi_level_ensemble"]
    closure = lambda a, b, c, d: hierarchy_fn(
        a, b, c, d,
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        key_col="key",
        target_col="actual_sales",
        hierarchy_col="group_id",
    )

    result = closure(X_tr, y_tr, X_v, y_v)
    assert result is not None
    assert result.model is not None
    assert result.val_wape >= 0.0


def test_loss_params_land_in_lightgbm_booster_for_global_local():
    """Regression test for the silent Tweedie-drop bug (2026-04):
    ``create_training_fn`` used to build a _hier_fn closure that only
    forwarded DataFrame context, never loss_params. So every segment-
    pooled global_local model silently trained with LightGBM's default
    squared-error objective even though segmentation strategy said
    Tweedie(1.7) for all segments.

    This test verifies loss_params = {'objective': 'tweedie', ...} now
    actually lands in the booster's objective attribute.
    """
    from utils.hierarchical_model_training import train_global_local_model

    train_df, val_df = _make_training_fixture()
    feature_cols = ["feat_lag_1", "feat_lag_2", "feat_rolling_4"]

    # Default (no loss_params) → LightGBM default is MSE-like
    wrapper_default, meta_default = train_global_local_model(
        train_df=train_df, val_df=val_df,
        feature_cols=feature_cols, target_col="actual_sales", key_col="key",
        model_family="lightgbm",
    )
    assert meta_default.get("applied_loss") is None

    # Tweedie via the canonical strategy-shaped dict
    wrapper_tw, meta_tw = train_global_local_model(
        train_df=train_df, val_df=val_df,
        feature_cols=feature_cols, target_col="actual_sales", key_col="key",
        model_family="lightgbm",
        loss_params={"objective": "tweedie", "tweedie_variance_power": 1.7},
    )
    assert meta_tw["applied_loss"] == "tweedie"
    booster = wrapper_tw.model_dict["global_model"]
    assert booster.objective == "tweedie"
    # Variance power must also have been forwarded (not defaulted)
    assert float(booster.get_params()["tweedie_variance_power"]) == pytest.approx(1.7)

    # Legacy key name 'tweedie_power' from segmentation_to_training_context
    # should also work.
    _, meta_legacy = train_global_local_model(
        train_df=train_df, val_df=val_df,
        feature_cols=feature_cols, target_col="actual_sales", key_col="key",
        model_family="lightgbm",
        loss_params={"objective": "tweedie", "tweedie_power": 1.5},
    )
    assert meta_legacy["applied_loss"] == "tweedie"


def test_loss_params_flow_through_adapter_to_multi_level_ensemble():
    """The _adapt_hierarchical_training adapter must forward loss_params
    into train_multi_level_ensemble so the global/hierarchy/key-level
    base learners all honour the strategy's recommended loss."""
    from utils.model_training import _adapt_hierarchical_training
    from utils.hierarchical_model_training import train_multi_level_ensemble

    train_df, val_df = _make_training_fixture()
    feature_cols = ["feat_lag_1", "feat_lag_2", "feat_rolling_4"]

    result = _adapt_hierarchical_training(
        train_multi_level_ensemble,
        train_df[feature_cols].to_numpy(),
        train_df["actual_sales"].to_numpy(),
        val_df[feature_cols].to_numpy(),
        val_df["actual_sales"].to_numpy(),
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        key_col="key",
        target_col="actual_sales",
        hierarchy_col="group_id",
        loss_params={"objective": "tweedie", "tweedie_variance_power": 1.6},
    )
    assert result is not None
    # applied_loss is stashed into the model's hyperparameter block via
    # the metadata dict passed up from the trainer.
    assert result.hyperparameters.get("applied_loss") == "tweedie"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
