"""Shared fixtures for the DE/UK segment-engine test suite."""
import sys
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DE_COMP = ROOT / "datacache" / "de" / "parallel_run" / "components"
DE_ACTUALS = ROOT / "datacache" / "de" / "parallel_run" / "actuals" / "actuals_202625.parquet"


@pytest.fixture
def de_cache_ready():
    """DE local cache (gitignored) — skip data-backed tests when it's absent (e.g. CI/Databricks)."""
    if not DE_COMP.exists() or not list(DE_COMP.glob("*.parquet")) or not DE_ACTUALS.exists():
        pytest.skip("DE datacache not present (local-only regression data)")
    return True


@pytest.fixture
def synth_components():
    """Synthetic component frame [key, year_week, s, c, rf, actual, category, segment, cs_gtin].

    Category A has two segments:
      * 'good'     — s == c == actual, rf = 1.5*actual (LEGO over-forecasts) -> the stack must beat LEGO.
      * 'legobest' — rf == actual, s == c == 0.5*actual                      -> LEGO wins, stack can't.
    Volumes are well above the 1000 floor so the guardrail (not the low-vol rule) decides.
    """
    weeks = ["202605", "202606", "202607", "202608"]
    rows = []
    for i, g in enumerate(["A_1", "A_2", "A_3", "A_4", "A_5", "A_6"]):   # 'good': 6 gtins
        for w in weeks:
            a = 200.0 + 20 * i
            rows.append(dict(key=f"{g}_acc", year_week=w, s=a, c=a, rf=1.5 * a, actual=a,
                             category="A", segment="good", cs_gtin=g))
    for i, g in enumerate(["B_1", "B_2", "B_3", "B_4", "B_5", "B_6"]):   # 'legobest': 6 gtins
        for w in weeks:
            a = 150.0 + 20 * i
            rows.append(dict(key=f"{g}_acc", year_week=w, s=0.5 * a, c=0.5 * a, rf=a, actual=a,
                             category="A", segment="legobest", cs_gtin=g))
    return pd.DataFrame(rows)
