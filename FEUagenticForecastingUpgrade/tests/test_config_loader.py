"""config_loader: .py preferred over .yaml; DE/UK base configs load with the keys the engine needs."""
import pathlib
import textwrap

import pytest

from utils.config_loader import load_market_config, load_config_file

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_de_config_loads_segment_keys():
    cfg = load_market_config("de")
    assert cfg["market_io"]["segment_col"] == "lego_segment"
    assert cfg["market_io"]["lego_pred_col"] == "prediction_xgb"
    assert cfg["target"] == "Actuals"
    assert "calibrate" in cfg["weight_fit"] and "guardrail_margin" in cfg["weight_fit"]


def test_uk_config_loads_segment_keys():
    cfg = load_market_config("uk")
    assert cfg["market_io"]["segment_col"] == "lego_segment"
    assert cfg["market_io"]["lego_pred_col"] == "prediction_rf"
    assert cfg["market_io"]["lego_subdir"] == "TF_pristine_allkeys"   # the fix (was TF_modelling_pristine)
    assert cfg["target"] == "actual_sales"


def test_uk_de_key_parity():
    """UK must be config-IDENTICAL to DE in structure (same market_io + weight_fit keys)."""
    de, uk = load_market_config("de"), load_market_config("uk")
    assert set(de["market_io"]) == set(uk["market_io"])
    assert set(de["weight_fit"]) == set(uk["weight_fit"])


def test_py_preferred_over_yaml(tmp_path):
    """When both exist, the .py wins; a lone .yaml still loads (Thailand / fallback)."""
    (tmp_path / "config_zz_base_local.py").write_text("CONFIG = {'target': 'from_py'}\n")
    (tmp_path / "config_zz_base_local.yaml").write_text("target: from_yaml\n")
    assert load_market_config("zz", str(tmp_path))["target"] == "from_py"
    (tmp_path / "config_zz_base_local.py").unlink()
    assert load_market_config("zz", str(tmp_path))["target"] == "from_yaml"


def test_load_config_file_prefers_py_sibling(tmp_path):
    """A .yaml path string resolves to its .py sibling (so legacy path-strings keep working)."""
    (tmp_path / "c.py").write_text("CONFIG = {'k': 1}\n")
    assert load_config_file(str(tmp_path / "c.yaml")) == {"k": 1}


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_market_config("nope", str(tmp_path))
