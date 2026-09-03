"""Unit tests for the rewritten classify_feature_availability (Phase A)."""

import numpy as np
import pandas as pd
import pytest

from config.schema import FeatureAvailabilityConfig
from utils.feature_availability import classify_feature_availability


def _build_df(n_keys=50, n_hist=100, n_future=20, seed=0):
    """Build a synthetic dataframe with a known set of columns for classification.

    Columns:
      - actual_sales: target
      - wthr_temp: dense continuous, present in history + future
      - price: dense continuous, present in history only
      - stale_nielsen: continuous, active in history but NOT in last 4 weeks
      - promo_monday: binary, 3% fill overall, but forward-planned
      - holiday_christmas_day: binary, fires only at week 52
      - event_one_off: binary, 0 everywhere in data (needs calendar synth)
      - rare_flag: binary, fires only in history, not in future
      - only_in_history_all: continuous, NO future presence
    """
    rng = np.random.default_rng(seed)
    hist_weeks = list(range(202401, 202401 + n_hist))
    fut_weeks = list(range(202401 + n_hist, 202401 + n_hist + n_future))
    all_weeks = hist_weeks + fut_weeks
    keys = [f"K{i:03d}" for i in range(n_keys)]

    rows = []
    for k in keys:
        for w in all_weeks:
            rows.append({"key": k, "year_week": w, "actual_sales": rng.integers(1, 100)})
    df = pd.DataFrame(rows)

    df["wthr_temp"] = rng.normal(15, 5, len(df))
    # Price: present in history, zero in future
    df["price"] = np.where(df["year_week"] <= hist_weeks[-1], rng.normal(5, 1, len(df)), 0.0)

    # Stale nielsen: present in history except the last 4 weeks, then zero everywhere
    last_4 = set(hist_weeks[-4:])
    df["stale_nielsen"] = np.where(
        (df["year_week"] <= hist_weeks[-1]) & (~df["year_week"].isin(last_4)),
        rng.normal(100, 10, len(df)), 0.0,
    )

    # Promo Monday: fires 3% of time, spread across 20% of keys, in both history and future
    active_keys = set(keys[: int(0.2 * n_keys)])
    is_active = df["key"].isin(active_keys).values
    promo_mask = rng.random(len(df)) < 0.03
    df["promo_monday"] = np.where(is_active & promo_mask, 1, 0).astype(np.int8)

    # Holiday Christmas Day: sparse; relies on pattern override (name contains 'holiday')
    last_week_of_year = df["year_week"] % 100 == 52
    df["holiday_christmas_day"] = np.where(last_week_of_year, 1, 0).astype(np.int8)

    # event_one_off: all zeros — pattern override should still save it
    df["event_one_off"] = 0

    # rare_flag: fires a few times in history, never in future
    df["rare_flag"] = 0
    df.loc[
        (df["year_week"].isin(hist_weeks[:10]))
        & (rng.random(len(df)) < 0.5),
        "rare_flag",
    ] = 1

    # Only-in-history continuous feature (truly history-only)
    df["only_in_history_all"] = np.where(df["year_week"] <= hist_weeks[-1],
                                          rng.normal(50, 5, len(df)), 0.0)
    return df, hist_weeks[-1]


def test_promo_rescued_by_pattern_override():
    df, cutoff = _build_df()
    res = classify_feature_availability(
        df=df, date_col="year_week", target_col="actual_sales",
        feature_cols=[c for c in df.columns if c not in ("key", "year_week", "actual_sales")],
        history_cutoff=str(cutoff), key_cols=["key"],
        fa_config=FeatureAvailabilityConfig(),
    )
    assert res["promo_monday"]["classification"] == "known_in_future", res["promo_monday"]
    assert res["promo_monday"]["pattern_matched"] is not None


def test_holiday_name_rescues_sparse_feature():
    df, cutoff = _build_df()
    res = classify_feature_availability(
        df=df, date_col="year_week", target_col="actual_sales",
        feature_cols=list(df.columns),
        history_cutoff=str(cutoff), key_cols=["key"],
        fa_config=FeatureAvailabilityConfig(),
    )
    assert res["holiday_christmas_day"]["classification"] == "known_in_future"


def test_zero_everywhere_event_still_rescued_by_pattern():
    """When a column name matches a pattern but has no signal, keep it as
    known_in_future anyway — calendar synth (or a human fix) can populate it."""
    df, cutoff = _build_df()
    res = classify_feature_availability(
        df=df, date_col="year_week", target_col="actual_sales",
        feature_cols=list(df.columns),
        history_cutoff=str(cutoff), key_cols=["key"],
        fa_config=FeatureAvailabilityConfig(),
    )
    assert res["event_one_off"]["classification"] == "known_in_future"


def test_stale_continuous_feature_excluded_by_recency_gate():
    df, cutoff = _build_df()
    cfg = FeatureAvailabilityConfig(
        recency_enabled=True, recency_lookback_periods=4,
        recency_min_active_keys=5,
    )
    res = classify_feature_availability(
        df=df, date_col="year_week", target_col="actual_sales",
        feature_cols=list(df.columns),
        history_cutoff=str(cutoff), key_cols=["key"],
        fa_config=cfg,
    )
    assert res["stale_nielsen"]["classification"] == "excluded"
    assert "stale" in res["stale_nielsen"]["reason"].lower() \
        or "recency" in res["stale_nielsen"]["reason"].lower() \
        or res["stale_nielsen"].get("recency_passed") is False


def test_dense_forward_feature_is_known_in_future():
    df, cutoff = _build_df()
    res = classify_feature_availability(
        df=df, date_col="year_week", target_col="actual_sales",
        feature_cols=list(df.columns),
        history_cutoff=str(cutoff), key_cols=["key"],
        fa_config=FeatureAvailabilityConfig(),
    )
    assert res["wthr_temp"]["classification"] == "known_in_future"


def test_history_only_continuous_with_recent_data_passes_recency():
    """`only_in_history_all` has values up to the cutoff — recency gate passes
    because the last 4 weeks of history are populated. It has no future
    presence so it should become history_only (NOT excluded)."""
    df, cutoff = _build_df()
    res = classify_feature_availability(
        df=df, date_col="year_week", target_col="actual_sales",
        feature_cols=list(df.columns),
        history_cutoff=str(cutoff), key_cols=["key"],
        fa_config=FeatureAvailabilityConfig(),
    )
    cls = res["only_in_history_all"]["classification"]
    assert cls in ("history_only", "partially_known"), (cls, res["only_in_history_all"])


def test_legacy_mode_back_compat_when_no_config():
    """Without fa_config, classifier falls back to the old fill-rate logic."""
    df, cutoff = _build_df()
    res = classify_feature_availability(
        df=df, date_col="year_week", target_col="actual_sales",
        feature_cols=list(df.columns),
        history_cutoff=str(cutoff), key_cols=["key"],
    )
    # Legacy classifier does not apply the recency gate. The dense forward
    # feature should still be known_in_future.
    assert res["wthr_temp"]["classification"] == "known_in_future"


def test_config_driven_pattern_override_is_configurable():
    """Users can drop 'promo_' from the pattern list and a sparse promo
    feature then falls through to indicator logic."""
    cfg = FeatureAvailabilityConfig(
        always_known_patterns=[p for p in FeatureAvailabilityConfig().always_known_patterns
                                if not p.startswith("promo")],
        indicator_min_future_nonzero_rows=1,
    )
    df, cutoff = _build_df()
    res = classify_feature_availability(
        df=df, date_col="year_week", target_col="actual_sales",
        feature_cols=list(df.columns),
        history_cutoff=str(cutoff), key_cols=["key"],
        fa_config=cfg,
    )
    # Still kept via auto-indicator detection (binary, with future presence).
    assert res["promo_monday"]["classification"] == "known_in_future"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
