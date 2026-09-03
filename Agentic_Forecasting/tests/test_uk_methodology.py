"""UK runs DE's IDENTICAL methodology — verified at the config/routing level (numbers need Databricks/UC)."""
from utils.config_loader import load_market_config
from utils.diq_runner import _load_market_io

UK_CATS = {"BEVERAGE", "CONDIMENT", "COOKING AID", "DEODORANT & FRAGRANCE", "FABRIC CLEANING",
           "FABRIC ENHANCER", "HAIR CARE", "HEALTH & WELLBEING", "HOME & HYGIENE", "MINI MEAL",
           "ORAL CARE", "OTH FOOD", "SKIN CARE", "SKIN CLEANSING"}


def test_uk_routes_to_segment_engine():
    """Setting market_io.segment_col is the gate (diq_runner.py) — confirm UK has it."""
    mio = _load_market_io("uk")
    assert mio.get("segment_col") == "lego_segment"


def test_uk_weight_fit_block_complete():
    wf = load_market_config("uk")["weight_fit"]
    for k in ("train_snapshots", "live_snapshot", "parent_dir_template", "lego_path_template",
              "actuals_source", "guardrail", "guardrail_margin", "calibrate", "in_scope_categories"):
        assert k in wf, f"UK weight_fit missing {k}"
    assert set(wf["in_scope_categories"]) == UK_CATS
    assert "TF_pristine_allkeys" in wf["lego_path_template"]   # the fix


def test_uk_and_de_same_methodology_surface():
    """UK and DE must expose the SAME config surface (keys) — identical methodology, different values."""
    de, uk = load_market_config("de"), load_market_config("uk")
    assert set(de["market_io"]) == set(uk["market_io"])
    assert set(de["weight_fit"]) == set(uk["weight_fit"])
    # both route to the segment engine
    assert de["market_io"]["segment_col"] and uk["market_io"]["segment_col"]


def test_uk_market_specifics():
    mio = load_market_config("uk")["market_io"]
    assert mio["lego_pred_col"] == "prediction_rf"            # UK benchmark
    assert mio["cats_subdir"] == "01_all_keys_dr_adjusted"    # UK panel
    assert mio["category_col"] == "category_name"
