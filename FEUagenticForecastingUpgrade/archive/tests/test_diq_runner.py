"""Tests for utils.diq_runner — the DIQ forward-forecast orchestrator.

These tests cover the helpers that run OUTSIDE the full ML pipeline:
  - category → YAML-template mapping (including the DEODORANT_FRAGRANCE override)
  - filter-by-category on a synthetic unified input table
  - permissive matching (token vs spaced-and-ampersand label)
  - error handling for absent columns / absent categories

The full end-to-end pipeline run is intentionally NOT tested here —
that's a ~30-min pipeline per category and belongs in integration
suites on Databricks. This file locks in the orchestration primitives
and shape guarantees.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from utils.diq_runner import (
    _category_to_yaml_slug,
    _filter_and_save_category,
    _yaml_template_for,
)


def test_category_slug_defaults_are_lowercase():
    assert _category_to_yaml_slug("COOKING_AID") == "cooking_aid"
    assert _category_to_yaml_slug("FABRIC_CLEANING") == "fabric_cleaning"
    assert _category_to_yaml_slug("hair_care") == "hair_care"


def test_category_slug_override_for_deo_fragrance():
    """DEODORANT_FRAGRANCE is the only slug that doesn't fall through the
    lowercase rule — the YAML calls it 'deo_fragrance'."""
    assert _category_to_yaml_slug("DEODORANT_FRAGRANCE") == "deo_fragrance"
    assert _category_to_yaml_slug("deodorant_fragrance") == "deo_fragrance"


@pytest.fixture
def project_root():
    return str(Path(__file__).resolve().parents[1])


def test_yaml_template_exists_for_every_uk_category(project_root):
    """Every UK category token used in the DIQ widget list must map to
    an existing template YAML. Guards against someone renaming or
    deleting a YAML without updating the override map."""
    uk_tokens = [
        "COOKING_AID", "CONDIMENT", "DEODORANT_FRAGRANCE",
        "FABRIC_CLEANING", "FABRIC_ENHANCER", "HAIR_CARE",
        "HOME_HYGIENE", "MINI_MEAL", "OTH_FOOD",
        "SKIN_CARE", "SKIN_CLEANSING",
    ]
    for t in uk_tokens:
        path = _yaml_template_for(t, project_root)
        assert path.exists(), f"Template missing for {t}: {path}"


def test_yaml_template_raises_on_unknown_category(project_root):
    with pytest.raises(FileNotFoundError, match="Template YAML not found"):
        _yaml_template_for("NOT_A_REAL_CATEGORY", project_root)


def _make_unified_input(n_per_cat=5):
    """Synthesise a unified input table with two categories."""
    rng = np.random.default_rng(0)
    rows = []
    for cat in ["COOKING_AID", "DEODORANT_FRAGRANCE"]:
        for i in range(n_per_cat):
            rows.append({
                "category_name": cat,
                "key": f"{cat}_{i:03d}",
                "year_week": f"20260{i+1}",
                "actual_sales": float(rng.integers(10, 500)),
            })
    return pd.DataFrame(rows)


def test_filter_and_save_category_writes_correct_subset(tmp_path):
    df = _make_unified_input(n_per_cat=5)
    target_dir = tmp_path / "sourcedata"
    path = _filter_and_save_category(
        full_df=df,
        category="COOKING_AID",
        category_col="category_name",
        sourcedata_dir=target_dir,
    )
    assert path.exists() and path.suffix == ".csv"
    back = pd.read_csv(path)
    assert len(back) == 5
    assert (back["category_name"] == "COOKING_AID").all()


def test_filter_and_save_accepts_spaced_label(tmp_path):
    """If the input table uses 'DEODORANT & FRAGRANCE' in the column
    (spaces + ampersand) but the widget token is 'DEODORANT_FRAGRANCE',
    the permissive matcher should still find the rows."""
    df = _make_unified_input(n_per_cat=3)
    df["category_name"] = df["category_name"].replace(
        "DEODORANT_FRAGRANCE", "DEODORANT & FRAGRANCE"
    )
    path = _filter_and_save_category(
        full_df=df,
        category="DEODORANT_FRAGRANCE",
        category_col="category_name",
        sourcedata_dir=tmp_path / "sourcedata",
    )
    back = pd.read_csv(path)
    assert len(back) == 3


def test_filter_and_save_raises_when_column_missing(tmp_path):
    df = _make_unified_input(n_per_cat=2)
    df = df.drop(columns=["category_name"])
    with pytest.raises(ValueError, match="no 'category_name' column"):
        _filter_and_save_category(
            full_df=df,
            category="COOKING_AID",
            category_col="category_name",
            sourcedata_dir=tmp_path / "sd",
        )


def test_filter_and_save_raises_when_category_absent(tmp_path):
    df = _make_unified_input(n_per_cat=2)
    with pytest.raises(ValueError, match="No rows match category"):
        _filter_and_save_category(
            full_df=df,
            category="NO_SUCH_CATEGORY",
            category_col="category_name",
            sourcedata_dir=tmp_path / "sd",
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
