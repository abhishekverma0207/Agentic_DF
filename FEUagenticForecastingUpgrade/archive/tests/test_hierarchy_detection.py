"""Unit tests for the rewritten hierarchy detector + resolver."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from utils.segmentation_enhanced import detect_hierarchy_chains
from utils.hierarchy_resolution import (
    HierarchyResolution,
    resolve_hierarchies,
    save_resolution,
)


# ======================================================================
# Synthetic fixtures
# ======================================================================

def _make_df(n_keys=100, n_weeks=20):
    """Synthetic data with a classic product hierarchy:

    market(2) → segment(5) → brand(10) → key(100), plus an orthogonal
    customer hierarchy APG_code(4) over the same keys.
    """
    rng = np.random.default_rng(0)
    keys = [f"K{i:03d}" for i in range(n_keys)]

    # Product hierarchy — nested
    brands = [f"B{i:02d}" for i in range(10)]
    segments = [f"S{i}" for i in range(5)]
    markets = [f"M{i}" for i in range(2)]

    key_to_brand = {k: brands[i % 10] for i, k in enumerate(keys)}
    brand_to_segment = {b: segments[i // 2] for i, b in enumerate(brands)}
    segment_to_market = {s: markets[i // 3] for i, s in enumerate(segments)}

    # Customer hierarchy — NOT nested with product
    key_to_apg = {k: f"APG{rng.integers(0, 4)}" for k in keys}

    # Binary indicator that would have fooled the old detector
    key_to_flag = {k: int(rng.integers(0, 2)) for k in keys}

    weeks = list(range(202401, 202401 + n_weeks))
    rows = []
    for k in keys:
        for w in weeks:
            rows.append({
                "key": k,
                "year_week": w,
                "actual_sales": float(rng.normal(100, 20)),
                "brand_name": key_to_brand[k],
                "segment_name": brand_to_segment[key_to_brand[k]],
                "market_name": segment_to_market[brand_to_segment[key_to_brand[k]]],
                "APG_code": key_to_apg[k],
                "ps_pos_retailers_mapped_keys": key_to_flag[k],
                "some_dense_float": float(rng.normal(5, 1)),  # not hierarchy
            })
    return pd.DataFrame(rows)


# ======================================================================
# Detector tests
# ======================================================================

def test_detector_finds_nested_product_chain():
    df = _make_df()
    res = detect_hierarchy_chains(
        df, ["key"], "year_week", "actual_sales",
        candidate_cols=["brand_name", "segment_name", "market_name", "APG_code"],
        min_keys_per_group=2, min_cardinality=2, include_binary=True,
    )
    product_cols = [it["column"] for it in res["product"]]
    # Expect coarse → fine order
    assert product_cols == ["market_name", "segment_name", "brand_name"], product_cols


def test_detector_separates_customer_chain():
    df = _make_df()
    res = detect_hierarchy_chains(
        df, ["key"], "year_week", "actual_sales",
        candidate_cols=["brand_name", "segment_name", "market_name", "APG_code"],
        min_keys_per_group=2, min_cardinality=2, include_binary=True,
    )
    customer_cols = [it["column"] for it in res["customer"]]
    assert customer_cols == ["APG_code"]


def test_detector_rejects_binary_indicator_by_default():
    df = _make_df()
    res = detect_hierarchy_chains(
        df, ["key"], "year_week", "actual_sales",
        candidate_cols=["brand_name", "ps_pos_retailers_mapped_keys"],
        min_keys_per_group=2,
        # default: include_binary=False, min_cardinality=3
    )
    rejected_cols = {r["column"] for r in res["candidates_rejected"]}
    assert "ps_pos_retailers_mapped_keys" in rejected_cols


def test_detector_with_include_binary_accepts_two_level():
    df = _make_df()
    res = detect_hierarchy_chains(
        df, ["key"], "year_week", "actual_sales",
        candidate_cols=["ps_pos_retailers_mapped_keys"],
        min_keys_per_group=2, min_cardinality=2, include_binary=True,
    )
    # Binary with 2 cardinalities allowed now → appears in 'other' (no name match)
    cols = (
        [it["column"] for it in res["product"]]
        + [it["column"] for it in res["customer"]]
        + [it["column"] for it in res["other"]]
    )
    assert "ps_pos_retailers_mapped_keys" in cols


def test_detector_collapses_duplicate_columns():
    df = _make_df().copy()
    # Aliases of segment_name
    df["form_name"] = df["segment_name"]
    df["sub_form_name"] = df["segment_name"]
    res = detect_hierarchy_chains(
        df, ["key"], "year_week", "actual_sales",
        candidate_cols=["segment_name", "form_name", "sub_form_name"],
        min_keys_per_group=2, min_cardinality=2,
    )
    # segment_name wins the collapse (matches segment pattern)
    assert "form_name" in res["duplicates_collapsed"]
    assert "sub_form_name" in res["duplicates_collapsed"]
    # Only one representative remains
    all_cols = sum([
        [it["column"] for it in res["product"]],
        [it["column"] for it in res["customer"]],
        [it["column"] for it in res["other"]],
    ], [])
    assert all_cols.count("segment_name") == 1
    assert "form_name" not in all_cols
    assert "sub_form_name" not in all_cols


def test_detector_warns_but_accepts_missing_column():
    df = _make_df()
    res = detect_hierarchy_chains(
        df, ["key"], "year_week", "actual_sales",
        candidate_cols=["brand_name", "does_not_exist"],
        min_keys_per_group=2, min_cardinality=2,
    )
    rejected_cols = {r["column"] for r in res["candidates_rejected"]}
    assert "does_not_exist" in rejected_cols


def test_detector_legacy_mode_with_no_candidates():
    df = _make_df()
    res = detect_hierarchy_chains(
        df, ["key"], "year_week", "actual_sales",
        candidate_cols=None,  # legacy / auto
        min_keys_per_group=2, min_cardinality=3,  # binary blocked
    )
    # Binary indicator excluded by default
    all_cols = sum([
        [it["column"] for it in res["product"]],
        [it["column"] for it in res["customer"]],
        [it["column"] for it in res["other"]],
    ], [])
    assert "ps_pos_retailers_mapped_keys" not in all_cols


# ======================================================================
# Resolver tests
# ======================================================================

def _fake_cfg_with_candidates(candidates):
    class _D:
        class _HD:
            def __init__(self):
                self.candidate_cols = candidates
                self.min_keys_per_group = 2
                self.min_cardinality = 2
                self.max_cardinality = 2000
                self.require_coverage = 0.95
                self.include_binary = True
                self.product_name_patterns = ["brand", "segment", "market"]
                self.customer_name_patterns = ["apg"]
                self.collapse_duplicate_columns = True
                self.max_depth_per_chain = 5
        def __init__(self):
            self.hierarchy_cols = []
            self.hierarchy_detection = self._HD()
    class _Cfg:
        def __init__(self):
            self.design = _D()
            self.prediction_key_cols = ["key"]
            self.timestamp_col = "year_week"
            self.target_col = "actual_sales"
    return _Cfg()


def test_resolver_tier3_candidate_list_runs_detector():
    df = _make_df()
    cfg = _fake_cfg_with_candidates(
        ["brand_name", "segment_name", "market_name", "APG_code"]
    )
    res = resolve_hierarchies(cfg, source_df=df)
    assert res.source == "candidate_list"
    assert res.product == ["market_name", "segment_name", "brand_name"]
    assert res.customer == ["APG_code"]
    # primary is score-based — for this synthetic data (brand=10,
    # segment=5, market=2, 100 keys total) brand_name has the best
    # cardinality / keys-per-group balance, so it's the expected pick.
    assert res.primary_product_col in ("brand_name", "segment_name")


def test_resolver_persists_and_reads_back():
    df = _make_df()
    cfg = _fake_cfg_with_candidates(
        ["brand_name", "segment_name", "market_name", "APG_code"]
    )
    with tempfile.TemporaryDirectory() as td:
        res = resolve_hierarchies(cfg, source_df=df, seg_dir=td)
        save_resolution(res, td)
        # Second call uses artifact (no source_df passed in)
        res2 = resolve_hierarchies(cfg, source_df=None, seg_dir=td)
        assert res2.source == "artifact"
        assert res2.product == res.product
        assert res2.customer == res.customer


def test_resolver_config_override_wins():
    df = _make_df()
    cfg = _fake_cfg_with_candidates(
        ["brand_name", "segment_name", "market_name", "APG_code"]
    )
    # Override via design.hierarchy_cols (list form)
    cfg.design.hierarchy_cols = ["brand_name"]
    res = resolve_hierarchies(cfg, source_df=df)
    assert res.source == "config_explicit"
    assert res.product == ["brand_name"]


def test_resolver_config_override_dict_form():
    df = _make_df()
    cfg = _fake_cfg_with_candidates(
        ["brand_name", "segment_name", "market_name", "APG_code"]
    )
    cfg.design.hierarchy_cols = {
        "product":  ["market_name", "brand_name"],
        "customer": ["APG_code"],
    }
    res = resolve_hierarchies(cfg, source_df=df)
    assert res.source == "config_explicit"
    assert res.product == ["market_name", "brand_name"]
    assert res.customer == ["APG_code"]
    assert res.primary_product_col == "brand_name"


def test_resolver_no_data_no_artifact_empty():
    cfg = _fake_cfg_with_candidates([])
    res = resolve_hierarchies(cfg, source_df=None, seg_dir=None)
    assert res.source == "empty"
    assert res.is_empty()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
