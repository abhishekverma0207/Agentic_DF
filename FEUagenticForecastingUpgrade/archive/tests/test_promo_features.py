"""Unit tests for utils.promo_features — leakage-free rolling promo lift."""

import numpy as np
import pandas as pd
import pytest

from utils.promo_features import create_promo_lift_features


def _make_df_single_key(n=40, promo_lift_true=2.0, seed=0):
    """Single-key fixture where historical on-promo sales average 2x non-promo."""
    rng = np.random.default_rng(seed)
    weeks = [202400 + i for i in range(n)]
    # alternating-ish promo pattern
    promo = np.array([(i % 3 == 0) for i in range(n)], dtype=int)
    base_sales = rng.normal(100.0, 5.0, size=n)
    sales = base_sales * np.where(promo == 1, promo_lift_true, 1.0)
    return pd.DataFrame({
        "key": "K1",
        "year_week": weeks,
        "actual_sales": sales,
        "promo_Monday": promo,
    })


def test_no_leakage_first_row_has_neutral_lift():
    df = _make_df_single_key(n=40)
    out = create_promo_lift_features(
        df, key_cols=["key"], target_col="actual_sales",
        date_col="year_week", forecast_lag=5,
    )
    # Row 0 has NO history available → lift must be 1.0 and any_history=0
    assert out.iloc[0]["promo_lift"] == 1.0
    assert out.iloc[0]["promo_lift_any_history"] == 0


def test_lift_converges_to_true_value_after_enough_history():
    df = _make_df_single_key(n=80, promo_lift_true=2.0)
    out = create_promo_lift_features(
        df, key_cols=["key"], target_col="actual_sales",
        date_col="year_week", forecast_lag=5, min_history_periods=8,
    )
    # After plenty of history the lifetime lift should converge near 2.0
    late = out.tail(10)["promo_lift"].mean()
    assert 1.5 < late < 2.5, f"expected lift ~2.0 after warm-up, got {late}"


def test_leakage_free_shift_excludes_recent_weeks():
    """If we shift by forecast_lag=5, the running stats at row i must NOT
    include data from rows i-4, i-3, i-2, i-1, i.
    """
    df = _make_df_single_key(n=30)
    out = create_promo_lift_features(
        df, key_cols=["key"], target_col="actual_sales",
        date_col="year_week", forecast_lag=5, min_history_periods=1,
    )
    # Manually compute expected sample size for the last row: all rows
    # 0..n-6 contribute; rows n-5..n-1 are shifted out. Any_promo series
    # has 10 promo rows (n//3) and 20 off-promo rows. After shifting by
    # 5, history at row 29 sees rows 0..24 inclusive.
    n = 30
    promo = np.array([(i % 3 == 0) for i in range(n)], dtype=int)
    on_hist = int(promo[:n - 5].sum())
    off_hist = int((1 - promo[:n - 5]).sum())
    expected_sample = min(on_hist, off_hist)
    assert out.iloc[-1]["promo_lift_sample_size"] == expected_sample


def test_multiple_keys_isolated():
    rng = np.random.default_rng(0)
    rows = []
    for k in ("A", "B", "C"):
        for i in range(30):
            promo = int(i % 2 == 0)
            sales = 100.0 if not promo else 300.0   # 3x lift
            sales += rng.normal(0, 1)
            rows.append({"key": k, "year_week": 202400 + i,
                         "actual_sales": sales, "promo_Monday": promo})
    df = pd.DataFrame(rows)
    out = create_promo_lift_features(
        df, ["key"], "actual_sales", "year_week",
        forecast_lag=5, min_history_periods=4,
    )
    # Each key independently converges to ~3
    for k in ("A", "B", "C"):
        last = out.loc[out["key"] == k].tail(5)["promo_lift"].mean()
        assert 2.5 < last < 3.5, (k, last)


def test_clipping_protects_from_extreme_lifts():
    # Pathological: baseline is tiny (non-promo sales ~0), promo is huge
    df = pd.DataFrame({
        "key": "K1",
        "year_week": [202400 + i for i in range(20)],
        "actual_sales": [0.01 if i % 2 == 0 else 1000.0 for i in range(20)],
        "promo_Monday": [1 if i % 2 else 0 for i in range(20)],
    })
    out = create_promo_lift_features(
        df, ["key"], "actual_sales", "year_week",
        forecast_lag=2, min_history_periods=4,
        lift_clip=(0.2, 5.0),
    )
    tail_lift = out.tail(5)["promo_lift"].max()
    assert tail_lift <= 5.0, f"clip not honoured: {tail_lift}"


def test_auto_detects_binary_promo_cols_and_ignores_continuous():
    df = _make_df_single_key(n=20).copy()
    df["promo_feature_Dump_Bins"] = np.random.choice([0, 1], size=20)
    df["promo_cash_discount_per_case_on_invoice"] = np.random.rand(20) * 10  # continuous
    df["promo_feature_Online"] = 0  # all zeros — still binary-valid
    out = create_promo_lift_features(
        df, ["key"], "actual_sales", "year_week", forecast_lag=5,
    )
    # Just verifying it runs and emits the four columns
    for col in ("promo_lift", "promo_lift_sample_size",
                "promo_lift_any_history", "promo_lift_recent_13w"):
        assert col in out.columns


def test_neutral_lift_when_no_flags_present():
    df = pd.DataFrame({
        "key": "K1",
        "year_week": [202400 + i for i in range(10)],
        "actual_sales": np.ones(10) * 100,
    })
    out = create_promo_lift_features(
        df, ["key"], "actual_sales", "year_week", forecast_lag=5,
    )
    # All neutral
    assert (out["promo_lift"] == 1.0).all()
    assert (out["promo_lift_any_history"] == 0).all()


def test_recent_variant_lags_lifetime_during_regime_change():
    # First 20 weeks: no lift. Next 20: 3x lift.
    n = 40
    rng = np.random.default_rng(42)
    promo = np.array([(i % 2 == 1) for i in range(n)], dtype=int)
    sales = np.zeros(n)
    for i in range(n):
        mul = 3.0 if (i >= 20 and promo[i]) else (1.0 if promo[i] else 1.0)
        sales[i] = 100.0 * mul + rng.normal(0, 1)
    df = pd.DataFrame({
        "key": "K1",
        "year_week": [202400 + i for i in range(n)],
        "actual_sales": sales,
        "promo_Monday": promo,
    })
    out = create_promo_lift_features(
        df, ["key"], "actual_sales", "year_week",
        forecast_lag=5, min_history_periods=4, recent_window=13,
    )
    # At the end of the regime-change series, the recent-13w lift should
    # reflect the NEW (3x) regime more strongly than the lifetime lift.
    last = out.iloc[-1]
    assert last["promo_lift_recent_13w"] > last["promo_lift"], (
        f"recent {last['promo_lift_recent_13w']} not > lifetime {last['promo_lift']}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
