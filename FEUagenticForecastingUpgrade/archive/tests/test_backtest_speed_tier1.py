"""Unit tests for Tier-1 backtest speedups:

* SW1 — source_df is loaded once in ``run_rolling_origin_backtest`` and
  forwarded to ``run_single_origin_inference`` and
  ``regenerate_features_for_backtest_origin``.
* ME1 — ``optimise_dataframe_memory`` casts wide object columns to
  ``category`` and downcasts integers safely. No accidental mutation of
  the value semantics.
* SW2 — per-origin forecast files are written as Parquet when
  ``pyarrow`` is available, falling back to CSV otherwise; the combine
  step handles both.
"""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ======================================================================
# ME1 — dataframe_memory.optimise_dataframe_memory
# ======================================================================

from utils.dataframe_memory import optimise_dataframe_memory


def _make_large_df(n_rows=150_000):
    """Large enough to exceed the 100 MB min_bytes_threshold so the
    function actually runs rather than short-circuiting."""
    rng = np.random.default_rng(0)
    # Several wide object columns with low cardinality (good category
    # candidates), one high-cardinality text column that must NOT be
    # cast (> max_category_ratio), plus int and float columns.
    return pd.DataFrame({
        "low_card_brand": rng.choice(["A", "B", "C", "D", "E"], size=n_rows),
        "low_card_segment": rng.choice(["X", "Y", "Z"], size=n_rows),
        "medium_card_apg": rng.choice(
            [f"APG{i:03d}" for i in range(30)], size=n_rows),
        "high_card_free_text": [f"row_{i}_" + "x" * 20 for i in range(n_rows)],
        "int_col_small": rng.integers(0, 255, size=n_rows),
        "int_col_large": rng.integers(-1_000_000, 1_000_000, size=n_rows),
        "float_col": rng.normal(0, 1, size=n_rows).astype(np.float64),
        "year_week": rng.integers(202000, 202800, size=n_rows),
    })


def test_me1_casts_low_cardinality_objects_to_category():
    df = _make_large_df()
    before_low = df["low_card_brand"].dtype
    assert before_low == object
    df2, diag = optimise_dataframe_memory(df, log_prefix="test-ME1", min_bytes_threshold=0)
    assert df2["low_card_brand"].dtype.name == "category"
    assert df2["low_card_segment"].dtype.name == "category"
    assert df2["medium_card_apg"].dtype.name == "category"
    assert diag["n_cat"] >= 3


def test_me1_leaves_high_cardinality_object_alone():
    df = _make_large_df()
    df2, _ = optimise_dataframe_memory(df, log_prefix="test-ME1-hc", min_bytes_threshold=0)
    # high_card_free_text has cardinality ≈ n_rows, well above 0.5 ratio
    assert df2["high_card_free_text"].dtype == object


def test_me1_downcasts_small_ints():
    df = _make_large_df()
    before_small = df["int_col_small"].dtype
    assert str(before_small).startswith("int")
    df2, diag = optimise_dataframe_memory(df, log_prefix="test-ME1-int", min_bytes_threshold=0)
    # int_col_small [0..255] → fits in uint8
    new_small = df2["int_col_small"].dtype
    assert new_small.itemsize <= np.dtype(np.int32).itemsize, (
        f"expected integer downcast, got {new_small}"
    )
    assert diag["n_int_down"] >= 1


def test_me1_protect_cols_are_not_modified():
    df = _make_large_df()
    df2, _ = optimise_dataframe_memory(
        df, protect_cols=["low_card_brand", "year_week"],
        log_prefix="test-ME1-protect",
        min_bytes_threshold=0,
    )
    assert df2["low_card_brand"].dtype == object
    assert df2["year_week"].dtype == df["year_week"].dtype


def test_me1_value_semantics_preserved():
    """Casts must not change any observable values — rowwise equality
    must hold after optimisation."""
    df = _make_large_df(n_rows=150_000)
    df_orig = df.copy()
    df2, _ = optimise_dataframe_memory(df, log_prefix="test-ME1-values", min_bytes_threshold=0)
    # All values must survive the cast round-trip when rendered back to str
    for col in df_orig.columns:
        assert (df2[col].astype(df_orig[col].dtype) == df_orig[col]).all(), col


def test_me1_skips_small_df():
    """Under the byte threshold the function must be a no-op."""
    df = pd.DataFrame({"a": ["x"] * 100, "b": [1] * 100})
    df2, diag = optimise_dataframe_memory(df, log_prefix="test-ME1-small")
    assert diag.get("skipped") is True
    # unchanged dtypes
    assert df2["a"].dtype == object


def test_me1_returns_same_dataframe_object():
    """For efficiency the helper returns the same DataFrame instance
    (in-place cast), not a copy."""
    df = _make_large_df()
    df2, _ = optimise_dataframe_memory(df, log_prefix="test-ME1-ident", min_bytes_threshold=0)
    assert df2 is df


# ======================================================================
# SW1 — signatures accept source_df parameter
# ======================================================================

def test_sw1_run_single_origin_inference_accepts_source_df_kwarg():
    from utils.backtesting import run_single_origin_inference
    sig = inspect.signature(run_single_origin_inference)
    assert "source_df" in sig.parameters
    # Default must be None so existing callers are unaffected
    assert sig.parameters["source_df"].default is None


def test_sw1_regenerate_features_accepts_source_df_kwarg():
    from utils.backtesting import regenerate_features_for_backtest_origin
    sig = inspect.signature(regenerate_features_for_backtest_origin)
    assert "source_df" in sig.parameters
    assert sig.parameters["source_df"].default is None


def test_sw1_inner_call_forwards_source_df():
    """The call inside run_single_origin_inference must pass source_df
    through to regenerate_features_for_backtest_origin. We check this
    statically so nobody breaks the chain in a future edit.
    """
    import utils.backtesting as bt
    src = inspect.getsource(bt.run_single_origin_inference)
    # Find the regenerate_features call and ensure source_df is listed
    idx = src.find("regenerate_features_for_backtest_origin(")
    assert idx >= 0
    # Check a reasonable window after the open-paren
    call_src = src[idx:idx + 800]
    assert "source_df=source_df" in call_src, (
        "run_single_origin_inference must forward source_df into "
        "regenerate_features_for_backtest_origin — the chain was broken. "
        f"Got:\n{call_src}"
    )


# ======================================================================
# SW2 — Parquet preference for per-origin intermediates
# ======================================================================

def test_sw2_run_rolling_origin_backtest_prefers_parquet_when_available():
    """The run_rolling_origin_backtest body must reference both ``.parquet``
    and ``to_parquet``, with a graceful fallback to CSV. Static-source
    inspection — we don't want to execute the full backtest."""
    import utils.backtesting as bt
    src = inspect.getsource(bt.run_rolling_origin_backtest)
    assert "to_parquet" in src, "SW2 parquet path missing"
    assert "_intermediate_ext" in src or ".parquet" in src, "parquet format not selected"
    # Fallback to CSV must still exist for environments without pyarrow
    assert "to_csv" in src


def test_sw2_combine_handles_both_parquet_and_csv():
    """The combine step at the end must auto-detect extension so runs
    made with pyarrow installed and without it interoperate."""
    import utils.backtesting as bt
    src = inspect.getsource(bt.run_rolling_origin_backtest)
    assert "read_parquet" in src, "combine step missing parquet support"
    assert "read_csv" in src, "combine step missing csv support"


# ======================================================================
# Integration — helper round-trip on a realistic wide DataFrame
# ======================================================================

def test_me1_real_world_shape():
    """Mirror the structure of the UK CONDIMENT source on a smaller
    scale: many wide object columns, some numeric, some id-like."""
    rng = np.random.default_rng(42)
    n = 120_000
    cols = {
        "key": [f"K{i % 500:04d}" for i in range(n)],
        "year_week": rng.integers(202000, 202800, size=n),
        "actual_sales": rng.normal(100, 20, size=n),
        # 20 wide object-typed columns that are really low-cardinality
        # metadata — this is what dominates UK source memory.
        **{f"meta_{i}": rng.choice([f"v{j}" for j in range(20)], size=n)
           for i in range(20)},
        # One binary indicator (should be downcast to uint8)
        "promo_flag": rng.integers(0, 2, size=n),
    }
    df = pd.DataFrame(cols)
    before_mem = df.memory_usage(deep=True).sum()
    df2, diag = optimise_dataframe_memory(
        df, protect_cols=["key", "year_week", "actual_sales"],
        log_prefix="test-ME1-realshape",
        min_bytes_threshold=0,
    )
    after_mem = df2.memory_usage(deep=True).sum()
    # Expect a meaningful reduction (>= 40% of object-column weight)
    assert after_mem < before_mem
    saved_pct = (before_mem - after_mem) / before_mem * 100
    assert saved_pct >= 30, f"expected ≥30% saving, got {saved_pct:.1f}%"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
