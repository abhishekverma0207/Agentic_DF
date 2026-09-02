"""
UK rolling-origin evaluation vs the LEGO `prediction_rf` benchmark.

We are judged on **accuracy (1−WAPE)** and **|bias|** at horizons **lag 4 and lag 5**,
pooled across rolling snapshot origins, **per category**, over the **full
non-excluded key universe**. A candidate forecast WINS a (category, lag) cell iff
it beats `prediction_rf` on BOTH accuracy and |bias|.

Data comes from the local cache built by `scripts/uc_cache.py`:
  datacache/uk/benchmark/snapshot=YYYYWW/part-*.parquet   (Forecast rows; prediction_rf + predicted)
  datacache/uk/actuals/actuals_latest.parquet             (real actuals to 202619)

Horizon is derived from `year_week` (NOT the `forecast_step` column, which is only
populated for the 2 categories the old model covered): `lag = ISO-weeks(snapshot→year_week) + 1`.

Metric math mirrors utils/lego_eval.py::_acc_bias (pooled ratios of sums); the
join/grain/benchmark-source is UK-specific (lego_eval is hardcoded to Thailand/E1016).

Typical use
-----------
    from utils import uk_eval
    SNAPS = ["202608","202609","202610","202611","202612","202613","202614","202615"]

    # Sanity: where does the OLD model (`predicted`, 2 cats) stand vs the benchmark?
    fr = uk_eval.load_scoring_frame(SNAPS, lags=(4,5))
    sb = uk_eval.scoreboard(fr, cand_col="predicted", missing_cand="drop")

    # A new model's forecast (full universe) — penalise any missing key as 0:
    fr2 = uk_eval.attach_candidate(fr, my_forecast_df, value_col="forecast", name="mine")
    sb2 = uk_eval.scoreboard(fr2, cand_col="mine", missing_cand="zero")
"""
from __future__ import annotations

import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("datacache/uk")

# Mirror of scripts/uc_cache.py IN_SCOPE_CATEGORIES (kept local to avoid a
# utils->scripts import). Core FMCG, dropping the 4 non-core categories.
IN_SCOPE_CATEGORIES = {
    "BEVERAGE", "CONDIMENT", "COOKING AID", "DEODORANT & FRAGRANCE",
    "FABRIC CLEANING", "FABRIC ENHANCER", "HAIR CARE", "HEALTH & WELLBEING",
    "HOME & HYGIENE", "MINI MEAL", "ORAL CARE", "OTH FOOD",
    "SKIN CARE", "SKIN CLEANSING",
}

# Modelable rolling origins: 202609..202615 (7). 202608's INPUT panel is absent
# upstream (only DT/TF_pristine_allkeys exist), so we cannot forecast it — exclude
# it from the eval so coverage stays apples-to-apples with what the model produces.
DEFAULT_SNAPSHOTS = ["202609", "202610", "202611", "202612", "202613", "202614", "202615"]


# --------------------------------------------------------------------------- #
# horizon / metric primitives
# --------------------------------------------------------------------------- #
def _monday(yyyyww: str) -> datetime:
    s = str(yyyyww)
    y, w = int(s[:4]), int(s[4:])
    return datetime.strptime(f"{y}-W{w:02d}-1", "%G-W%V-%u")


def iso_step(year_week, snapshot) -> int:
    """Horizon/lag of `year_week` for a forecast made at `snapshot`.
    lag 1 = the snapshot week itself. e.g. iso_step(202612, 202609) == 4."""
    delta_days = (_monday(year_week) - _monday(snapshot)).days
    return delta_days // 7 + 1


def acc_bias(actual, fcst):
    """Return (accuracy=1−WAPE, bias, sum_actual) as pooled ratios of sums.
    Mirrors utils/lego_eval.py:30. NaNs are treated as 0."""
    a = np.nan_to_num(np.asarray(actual, dtype="float64"), nan=0.0)
    f = np.nan_to_num(np.asarray(fcst, dtype="float64"), nan=0.0)
    sa = a.sum()
    if sa <= 0:
        return np.nan, np.nan, sa
    return 1.0 - np.abs(f - a).sum() / sa, (f - a).sum() / sa, sa


# --------------------------------------------------------------------------- #
# cache readers
# --------------------------------------------------------------------------- #
def _read_cached_benchmark(snapshot: str) -> pd.DataFrame:
    parts = sorted(glob.glob(str(CACHE_ROOT / "benchmark" / f"snapshot={snapshot}" / "part-*.parquet")))
    if not parts:
        raise FileNotFoundError(
            f"No cached benchmark for snapshot {snapshot} under {CACHE_ROOT}/benchmark/snapshot={snapshot}/. "
            "Run: .venv/bin/python scripts/uc_cache.py --kind benchmark --snapshots "
            f"{snapshot[:4]}-{snapshot[4:]}"
        )
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def _read_actuals() -> pd.DataFrame:
    p = CACHE_ROOT / "actuals" / "actuals_latest.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"No cached actuals at {p}. Run: .venv/bin/python scripts/uc_cache.py --kind actuals"
        )
    a = pd.read_parquet(p)[["key", "year_week", "actual_sales"]].rename(columns={"actual_sales": "actual"})
    a["year_week"] = a["year_week"].astype(str)
    return a.drop_duplicates(["key", "year_week"])


# --------------------------------------------------------------------------- #
# scoring frame
# --------------------------------------------------------------------------- #
def load_scoring_frame(
    snapshots: Sequence[str] = DEFAULT_SNAPSHOTS,
    *,
    lags: Sequence[int] = (4, 5),
    cats: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the pooled long frame: one row per (snapshot, key, year_week) at the
    requested lags, non-excluded, in-scope, with `prediction_rf`, `predicted`,
    `New_FCST`, `lego_segment`, `horizon`, and joined real `actual`.
    """
    cats = set(cats) if cats is not None else IN_SCOPE_CATEGORIES
    acts = _read_actuals()
    lags = set(int(x) for x in lags)
    frames = []
    for s in snapshots:
        s = str(s)
        b = _read_cached_benchmark(s)
        b = b[b["category_name"].isin(cats)]
        b = b[pd.to_numeric(b["exclude_forecast"], errors="coerce") == 0].copy()
        b["year_week"] = b["year_week"].astype(str)
        yw_map = {yw: iso_step(yw, s) for yw in b["year_week"].unique()}
        b["horizon"] = b["year_week"].map(yw_map)
        b = b[b["horizon"].isin(lags)]
        b["snapshot"] = s
        frames.append(b)
    fr = pd.concat(frames, ignore_index=True)
    fr = fr.drop_duplicates(["snapshot", "key", "year_week"])
    fr = fr.merge(acts, on=["key", "year_week"], how="left")
    fr["actual"] = fr["actual"].fillna(0.0)
    fr["cs_gtin"] = fr["key"].astype(str).str.rsplit("_", n=1).str[0]   # GTIN-level grain
    logger.info("[uk_eval] scoring frame: %s rows | snapshots=%s | lags=%s | cats=%d",
                f"{len(fr):,}", list(snapshots), sorted(lags), len(cats))
    return fr


def attach_candidate(frame: pd.DataFrame, fc: pd.DataFrame, *, value_col: str, name: str = "cand") -> pd.DataFrame:
    """Left-join an external forecast onto the scoring frame.

    Joins on (snapshot, key, year_week) when both frames carry `snapshot` — the
    correct grain, since the same (key, year_week) is forecast by multiple origins
    at different horizons. Falls back to (key, year_week) for single-origin frames.
    `fc` must have key, year_week, <value_col> (and snapshot for the pooled case).
    """
    on = ["key", "year_week"]
    if "snapshot" in frame.columns and "snapshot" in fc.columns:
        on = ["snapshot", "key", "year_week"]
    f = fc[on + [value_col]].copy()
    f["year_week"] = f["year_week"].astype(str)
    if "snapshot" in on:
        f["snapshot"] = f["snapshot"].astype(str)
    f = f.drop_duplicates(on).rename(columns={value_col: name})
    return frame.merge(f, on=on, how="left")


# --------------------------------------------------------------------------- #
# scoreboard
# --------------------------------------------------------------------------- #
def _score(g: pd.DataFrame, cand_col: str, bench_col: str) -> dict:
    a = g["actual"].values
    acc_l, bias_l, sa = acc_bias(a, g[bench_col].values)
    acc_c, bias_c, _ = acc_bias(a, g[cand_col].values)
    return dict(
        n=len(g), sum_actual=float(sa),
        acc_cand=acc_c, acc_lego=acc_l, acc_gap=(acc_c - acc_l) if pd.notna(acc_c) and pd.notna(acc_l) else np.nan,
        bias_cand=bias_c, bias_lego=bias_l,
        acc_win=bool(acc_c > acc_l), bias_win=bool(abs(bias_c) < abs(bias_l)),
        WIN=bool((acc_c > acc_l) and (abs(bias_c) < abs(bias_l))),
    )


def scoreboard(
    frame: pd.DataFrame,
    cand_col: str,
    *,
    bench_col: str = "prediction_rf",
    missing_cand: str = "zero",
    level: str = "gtin",
) -> pd.DataFrame:
    """Per (category, lag) accuracy & |bias| of `cand_col` vs `bench_col`, with WIN flags.

    level:
      'gtin' (default, PRIMARY) — aggregate forecast & actual to cs_gtin × year_week
        (sum across customer keys) BEFORE computing error. Customer-level errors
        cancel, so this is higher and the decision-relevant grain.
      'key'  — score at the raw key (gtin×customer) × year_week grain.
    Both operate on the non-excluded universe carried in `frame`.

    Each lag (4, 5, …) is scored INDEPENDENTLY — never pooled. Adds `__ALL__`
    category roll-ups per lag.

    missing_cand:
      'zero' (default) — keys the candidate didn't forecast count as 0 (honest
        full-universe metric; use for the new model).
      'drop'           — score only rows where the candidate made a forecast
        (coverage-fair head-to-head; use for the old `predicted`, 2 cats only).
    """
    f = frame.copy()
    if missing_cand == "drop":
        f = f[f[cand_col].notna()]
    elif missing_cand == "zero":
        f[cand_col] = f[cand_col].fillna(0.0)
    else:
        raise ValueError("missing_cand must be 'zero' or 'drop'")
    if f.empty:
        return pd.DataFrame()
    f[bench_col] = f[bench_col].fillna(0.0)

    if level == "gtin":
        if "cs_gtin" not in f.columns:
            f["cs_gtin"] = f["key"].astype(str).str.rsplit("_", n=1).str[0]
        gcols = [c for c in ["snapshot", "cs_gtin", "year_week", "category_name", "horizon"] if c in f.columns]
        f = f.groupby(gcols, as_index=False).agg(
            actual=("actual", "sum"), **{bench_col: (bench_col, "sum"), cand_col: (cand_col, "sum")})
    elif level != "key":
        raise ValueError("level must be 'gtin' or 'key'")

    # Each lag is scored INDEPENDENTLY — no pooling across horizons.
    recs = []
    for (cat, h), g in f.groupby(["category_name", "horizon"]):
        recs.append({"category": cat, "lag": int(h), **_score(g, cand_col, bench_col)})
    for h, g in f.groupby("horizon"):
        recs.append({"category": "__ALL__", "lag": int(h), **_score(g, cand_col, bench_col)})

    sb = pd.DataFrame(recs)
    sb["level"] = level
    sb = sb.sort_values(["category", "lag"]).reset_index(drop=True)
    return sb


def win_summary(sb: pd.DataFrame) -> pd.DataFrame:
    """Compact view: per-category WIN at lag 4, lag 5, and combined."""
    real = sb[sb["category"] != "__ALL__"]
    piv = real.pivot_table(index="category", columns="lag", values="WIN", aggfunc="first")
    return piv


def evaluate(
    snapshots: Sequence[str] = DEFAULT_SNAPSHOTS,
    *,
    cand_col: str = "predicted",
    missing_cand: str = "drop",
    lags: Sequence[int] = (4, 5),
    cats: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Convenience: load + score a column already present in the benchmark
    (`predicted`, `New_FCST`, or `prediction_rf` itself). Returns the scoreboard."""
    fr = load_scoring_frame(snapshots, lags=lags, cats=cats)
    sb = scoreboard(fr, cand_col=cand_col, missing_cand=missing_cand)
    return sb


__all__ = [
    "CACHE_ROOT", "IN_SCOPE_CATEGORIES", "DEFAULT_SNAPSHOTS",
    "iso_step", "acc_bias", "load_scoring_frame", "attach_candidate",
    "scoreboard", "win_summary", "evaluate",
]
