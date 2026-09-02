#!/usr/bin/env python3
"""
Build a compact LOCAL parquet cache of UK forecasting data from Databricks UC.

Why
---
The UC sources are large Spark parquet directories (the input feature panel is
~12M rows x 294 cols/snapshot). We only need a curated slice for modelling +
evaluation. This tool mirrors just that slice to `datacache/uk/` as small,
snappy-compressed parquet so the whole build runs locally, then migrates to
Databricks.

Three kinds
-----------
- input     : per-snapshot modelling features (294 cols -> curated ~90).
              Path: .../DemandIQ/{YYYYWW}/UK/.../snapshot/{YYYY-WW}/01_all_keys_dr_adjusted_with_exclude
- benchmark : per-snapshot TF_DIQ_combined (42 cols -> ~12). Carries `prediction_rf`
              (LEGO benchmark, all cats) and `predicted` (ours, 2 cats). Forecast rows only.
- actuals   : the latest actuals (snapshot 2026-20, real actuals to 202619),
              trimmed to key/year_week/category_name/actual_sales.

Layout
------
    datacache/uk/
      input/      snapshot=YYYYWW/part-*.parquet
      benchmark/  snapshot=YYYYWW/part-*.parquet
      actuals/    actuals_latest.parquet
      manifest.json

Memory-bounded: parts are listed, then read ONE AT A TIME with column projection,
row-filtered, and written individually. A whole Spark dir is never loaded.

CLI
---
    .venv/bin/python scripts/uc_cache.py --kind input     --snapshots 2026-09
    .venv/bin/python scripts/uc_cache.py --kind benchmark  --snapshots 2026-08..2026-15
    .venv/bin/python scripts/uc_cache.py --kind actuals
    .venv/bin/python scripts/uc_cache.py --all             --snapshots 2026-08..2026-15
    # smoke test a couple of parts only:
    .venv/bin/python scripts/uc_cache.py --kind input --snapshots 2026-09 --max-parts 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import uc_io  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("uc_cache")

CACHE_ROOT = Path("datacache/uk")

# In-scope categories: all core FMCG, dropping the 4 non-core ones.
IN_SCOPE_CATEGORIES = {
    "BEVERAGE", "CONDIMENT", "COOKING AID", "DEODORANT & FRAGRANCE",
    "FABRIC CLEANING", "FABRIC ENHANCER", "HAIR CARE", "HEALTH & WELLBEING",
    "HOME & HYGIENE", "MINI MEAL", "ORAL CARE", "OTH FOOD",
    "SKIN CARE", "SKIN CLEANSING",
}
DROPPED_CATEGORIES = {
    "ICE CREAM CATEGORY", "NON CORPORATE PC CATEGORY",
    "PROFESSIONAL CLEANING CATEGORY", "PLANT BASED MEAT",
}

# Curated INPUT feature columns (drops the ~90-col promo_instore_* mirror and the
# ~40 sparse secondary_mechanic one-hots). Names intersected with the actual
# schema at read time, so a missing name is skipped (warned), never fatal.
INPUT_COLS: List[str] = [
    # ids / hierarchy
    "key", "year_week", "cs_gtin", "category_name", "APG_code", "APG_description",
    "planning_customer_code", "brand_name", "sub_sector_name", "segment_name",
    "form_name", "size_pack_name", "variant_name", "forecast_familygroup",
    "year_quarter",
    # target / dispatch / smoothing
    "actual_sales", "expected_dispatch_quantity", "actual_dispatch_quantity",
    "dispatch_rate", "avg_dispatch_rate_lego", "avg_dispatch_rate_lego_lag1",
    "smooth_sales", "smooth_sales_dispatch_adj", "smooth_sales_dr_sales_cap",
    "weeks_post_significant_dr_drop", "dr_flag_threshold",
    # POS / Nielsen
    "nielsen_pos_sales_cases", "nielsen_pos_sales_units",
    "nielsen_pos_price_per_case_weighted_avg", "nielsen_pos_numeric_dist_weighted_avg",
    "nielsen_pos_acvweighted_dist_weighted_avg",
    "retailer_pos_sales_cases", "retailer_pos_sales_units",
    "retailer_pos_price_per_case_weighted_avg", "retailer_pos_numeric_dist_weighted_avg",
    "retailer_pos_acvweighted_dist_weighted_avg",
    "primary_sales_for_latest_52_weeks", "retailer_pos_sales_cases_for_latest_52_weeks",
    "nielsen_pos_sales_cases_for_latest_52_weeks",
    "ratio_of_ps_to_retailer_pos_for_latest_52_weeks",
    "ratio_of_ps_to_nielsen_pos_for_latest_52_weeks",
    # pricing
    "gsv_price_per_case_weighted_avg", "niv_price_per_case_weighted_avg",
    "pricing_list_price_weighted_avg", "new_price_pricing_list_price_weighted_avg",
    # promo (curated shipment block)
    "promo_shipment_flag", "promo_num_days_shipped", "promo_planned_volume",
    "promo_planned_volume_total", "promo_cash_discount_per_case_off_invoice",
    "promo_cash_discount_per_case_on_invoice",
    "promo_primary_mechanic_Special_Packs_Offer", "promo_primary_mechanic_Other",
    "promo_primary_mechanic_Shopper_Marketing", "promo_primary_mechanic_EDLP",
    "promo_primary_mechanic_MultiBuy", "promo_primary_mechanic_TPR",
    "promo_primary_mechanic_Loyalty", "promo_primary_mechanic_WinterSummer_Wall",
    "promo_primary_mechanic_Pipe_Fill", "promo_summer_winter_wall_flag",
    "promo_pipefill_flag", "promo_WIGIG_flag", "promo_feature_pallet_drop_flag",
    "promo_instore_flag",
    # calendar / weather
    "holiday_flag", "holiday_easter_flag", "pancake_holiday_flag",
    "august_bank_holiday_flag", "season_summer", "season_winter", "season_spring",
    "season_autumn", "high_sunshine", "WTHR_avgTemp", "WTHR_avgTempVsNorm",
    "ice_ooh_month_end_flag",
    # lifecycle / supply
    "delist_flag", "delist_week", "transition_flag", "UoM_change_flag",
    "black_swan_ind", "supply_constraints_ind", "customer_retaliation_ind",
    "one_off_anamolies_ind", "gap_filling_ind", "min_yw_key",
    # exclusion
    "exclude_forecast",
]

# Benchmark: just what eval/diagnosis needs. `actual`/`forecast_step`/`lag` kept
# where present (only populated for the 2 `predicted` cats) but we derive horizon
# from year_week in the evaluator, so they're informational.
BENCHMARK_COLS: List[str] = [
    "key", "year_week", "category_name", "Period_Type", "predicted",
    "prediction_rf", "New_FCST", "exclude_forecast", "lego_segment",
    "forecast_step", "lag", "actual_sales",
]

ACTUALS_COLS: List[str] = ["key", "year_week", "category_name", "actual_sales"]

# Real actuals are known up to this week (later weeks in the 2026-20 panel are placeholders).
ACTUALS_MAX_YW = "202619"
ACTUALS_SNAPSHOT = "202620"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def add_weeks(yyyyww: str, n: int) -> str:
    """ISO-week-correct add (handles 52/53-week years and year rollover)."""
    y, w = int(yyyyww[:4]), int(yyyyww[4:])
    monday = datetime.strptime(f"{y}-W{w:02d}-1", "%G-W%V-%u")
    iso = (monday + pd.Timedelta(weeks=n)).isocalendar()
    return f"{iso[0]}{iso[1]:02d}"


def _compact(s: str) -> str:
    return s.strip().replace("-", "")


def expand_snapshots(spec: str) -> List[str]:
    """'2026-08..2026-15' -> [202608..202615]; '2026-09,2026-12' -> [202609,202612]."""
    spec = spec.strip()
    if ".." in spec:
        a, b = (_compact(x) for x in spec.split("..", 1))
        out, cur = [], a
        for _ in range(200):
            out.append(cur)
            if cur == b:
                break
            cur = add_weeks(cur, 1)
        return out
    return [_compact(x) for x in spec.split(",") if x.strip()]


def _input_path(yyyyww: str) -> str:
    dash = f"{yyyyww[:4]}-{yyyyww[4:]}"
    return (f"/Volumes/pds_feu_931272_dev/eu_uk/staging/DemandIQ/{yyyyww}/UK/"
            f"lego_runs/forward_run/total_forecast/snapshot/{dash}/"
            f"01_all_keys_dr_adjusted_with_exclude")


def _benchmark_path(yyyyww: str) -> str:
    dash = f"{yyyyww[:4]}-{yyyyww[4:]}"
    return f"/Volumes/pds_feu_931272_dev/data_science_team/uk_run/OUTPUTS/TF_DIQ_combined_{dash}"


def _actuals_path() -> str:
    dash = f"{ACTUALS_SNAPSHOT[:4]}-{ACTUALS_SNAPSHOT[4:]}"
    return (f"/Volumes/pds_feu_931272_dev/eu_uk/staging/DemandIQ/{ACTUALS_SNAPSHOT}/UK/"
            f"lego_runs/forward_run/total_forecast/snapshot/{dash}/01_all_keys_dr_adjusted")


def _list_parts(src: str, profile: str) -> List[str]:
    names = uc_io.ls(src, profile=profile)
    parts = sorted(n for n in names if n.startswith("part-") and n.endswith((".parquet", ".pq")))
    return [f"{src}/{n}" for n in parts]


def _in_scope(df: pd.DataFrame) -> pd.DataFrame:
    if "category_name" in df.columns:
        df = df[df["category_name"].isin(IN_SCOPE_CATEGORIES)]
    if "key" in df.columns:
        df = df[df["key"].notna()]
    return df


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def _manifest_path() -> Path:
    return CACHE_ROOT / "manifest.json"


def _update_manifest(entry_key: str, info: dict) -> None:
    mp = _manifest_path()
    data = {}
    if mp.exists():
        try:
            data = json.loads(mp.read_text())
        except Exception:
            data = {}
    data[entry_key] = info
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(data, indent=2, sort_keys=True))


# --------------------------------------------------------------------------- #
# core: stream parts -> trimmed parquet
# --------------------------------------------------------------------------- #
def cache_parts(
    src: str,
    out_dir: Path,
    want_cols: Optional[List[str]],     # None => keep ALL columns (full-input mode)
    row_filter: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    profile: str,
    max_parts: Optional[int] = None,
    overwrite: bool = False,
) -> dict:
    parts = _list_parts(src, profile)
    if max_parts is not None:
        parts = parts[:max_parts]
    if not parts:
        raise FileNotFoundError(f"No part files under {src!r}")
    out_dir.mkdir(parents=True, exist_ok=True)

    keep: Optional[List[str]] = None
    missing: List[str] = []
    n_rows = 0
    logger.info("  %d parts in %s", len(parts), src)
    for i, part in enumerate(parts):
        dst = out_dir / f"part-{i:05d}.parquet"
        if dst.exists() and not overwrite:
            try:
                n_rows += len(pd.read_parquet(dst, columns=["key"]))
            except Exception:
                pass
            continue
        if keep is None:
            # First part: read full to learn the real schema, then project.
            df = uc_io.read_parquet(part, profile=profile)
            avail = list(df.columns)
            if want_cols is None:                            # full-input mode
                keep = avail
                missing = []
            else:
                aset = set(avail)
                keep = [c for c in want_cols if c in aset]
                missing = [c for c in want_cols if c not in aset]
                if missing:
                    logger.warning("  %d requested cols absent (skipped): %s",
                                   len(missing), ", ".join(missing[:8]) + ("…" if len(missing) > 8 else ""))
                df = df[keep]
        else:
            df = uc_io.read_parquet(part, columns=keep, profile=profile)
        df = row_filter(df)
        df.to_parquet(dst, index=False)
        n_rows += len(df)
        if (i + 1) % 5 == 0 or i == len(parts) - 1:
            logger.info("  part %d/%d -> %s rows so far", i + 1, len(parts), f"{n_rows:,}")

    return {
        "src": src,
        "out_dir": str(out_dir),
        "n_parts": len(parts),
        "n_rows": n_rows,
        "cols_kept": keep if keep is not None else want_cols,
        "cols_missing": missing,
        "bytes": sum(p.stat().st_size for p in out_dir.glob("part-*.parquet")),
        "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# per-kind drivers
# --------------------------------------------------------------------------- #
def cache_input(yyyyww: str, *, profile: str, max_parts=None, overwrite=False, full: bool = False) -> dict:
    cutoff = add_weeks(yyyyww, 12)  # keep history + the 13-week forecast window only

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = _in_scope(df)
        if "year_week" in df.columns:
            df = df[df["year_week"].astype(str) <= cutoff]
        return df

    subdir = "input_full" if full else "input"
    cols = None if full else INPUT_COLS
    logger.info("[input%s] snapshot %s (keep year_week <= %s)",
                " FULL" if full else "", yyyyww, cutoff)
    info = cache_parts(_input_path(yyyyww), CACHE_ROOT / subdir / f"snapshot={yyyyww}",
                       cols, _filter, profile=profile, max_parts=max_parts, overwrite=overwrite)
    info["snapshot"] = yyyyww
    info["cutoff_year_week"] = cutoff
    info["full"] = full
    _update_manifest(f"input{'_full' if full else ''}:{yyyyww}", info)
    logger.info("[input%s] %s done: %s rows, %.1f MB", " FULL" if full else "",
                yyyyww, f"{info['n_rows']:,}", info["bytes"] / 1e6)
    return info


def cache_benchmark(yyyyww: str, *, profile: str, max_parts=None, overwrite=False) -> dict:
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = _in_scope(df)
        if "Period_Type" in df.columns:        # forecast rows only — eval needs these
            df = df[df["Period_Type"] == "Forecast"]
        return df

    logger.info("[benchmark] snapshot %s (Forecast rows, in-scope cats)", yyyyww)
    info = cache_parts(_benchmark_path(yyyyww), CACHE_ROOT / "benchmark" / f"snapshot={yyyyww}",
                       BENCHMARK_COLS, _filter, profile=profile, max_parts=max_parts, overwrite=overwrite)
    info["snapshot"] = yyyyww
    _update_manifest(f"benchmark:{yyyyww}", info)
    logger.info("[benchmark] %s done: %s rows, %.1f MB", yyyyww, f"{info['n_rows']:,}", info["bytes"] / 1e6)
    return info


def cache_actuals(*, profile: str, max_parts=None, overwrite=False) -> dict:
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = _in_scope(df)
        if "year_week" in df.columns:
            df = df[df["year_week"].astype(str) <= ACTUALS_MAX_YW]
        if "actual_sales" in df.columns:
            df = df[df["actual_sales"].notna()]
        return df

    logger.info("[actuals] from snapshot %s (year_week <= %s)", ACTUALS_SNAPSHOT, ACTUALS_MAX_YW)
    out_dir = CACHE_ROOT / "actuals"
    info = cache_parts(_actuals_path(), out_dir, ACTUALS_COLS, _filter,
                       profile=profile, max_parts=max_parts, overwrite=overwrite)
    # Consolidate the many small parts into one tidy file (actuals are small).
    parts = sorted(out_dir.glob("part-*.parquet"))
    if parts:
        full = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
        full = full.drop_duplicates(["key", "year_week"])
        full.to_parquet(out_dir / "actuals_latest.parquet", index=False)
        for p in parts:
            p.unlink()
        info["n_rows"] = len(full)
        info["bytes"] = (out_dir / "actuals_latest.parquet").stat().st_size
        info["consolidated"] = "actuals_latest.parquet"
    _update_manifest("actuals:latest", info)
    logger.info("[actuals] done: %s rows, %.1f MB", f"{info['n_rows']:,}", info["bytes"] / 1e6)
    return info


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Cache UK forecasting data from UC -> datacache/uk/")
    ap.add_argument("--kind", choices=["input", "benchmark", "actuals"],
                    help="Which source to cache. Omit with --all to do input+benchmark+actuals.")
    ap.add_argument("--all", action="store_true", help="Cache input+benchmark for --snapshots, plus actuals.")
    ap.add_argument("--snapshots", default="", help="e.g. '2026-09' or '2026-08..2026-15' or '2026-09,2026-12'")
    ap.add_argument("--profile", default=uc_io.DEFAULT_PROFILE)
    ap.add_argument("--max-parts", type=int, default=None, help="Cap parts per dir (smoke testing).")
    ap.add_argument("--overwrite", action="store_true", help="Re-download parts even if cached.")
    ap.add_argument("--full", action="store_true", help="--kind input: cache ALL 294 columns to datacache/uk/input_full/")
    a = ap.parse_args()

    snaps = expand_snapshots(a.snapshots) if a.snapshots else []
    kw = dict(profile=a.profile, max_parts=a.max_parts, overwrite=a.overwrite)

    def _safe(fn, *args, **kwargs):
        """Run a cache step; on a missing source dir (or any error) log + continue so
        one absent snapshot (e.g. 202608 has no 01_all_keys_dr_adjusted) doesn't abort the rest."""
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as exc:
            logger.warning("  SKIP %s%s — %s", fn.__name__, args, exc)
        except Exception as exc:  # incl. databricks NotFound
            if "not found" in str(exc).lower() or exc.__class__.__name__ == "NotFound":
                logger.warning("  SKIP %s%s — source dir not found", fn.__name__, args)
            else:
                logger.error("  FAIL %s%s — %s", fn.__name__, args, exc)
        return None

    if a.all:
        if not snaps:
            ap.error("--all needs --snapshots")
        for s in snaps:
            _safe(cache_input, s, **kw)
            _safe(cache_benchmark, s, **kw)
        _safe(cache_actuals, **kw)
    elif a.kind == "actuals":
        _safe(cache_actuals, **kw)
    elif a.kind in ("input", "benchmark"):
        if not snaps:
            ap.error(f"--kind {a.kind} needs --snapshots")
        fn = cache_input if a.kind == "input" else cache_benchmark
        for s in snaps:
            _safe(fn, s, **{**kw, **({"full": a.full} if a.kind == "input" else {})})
    else:
        ap.error("pass --kind {input,benchmark,actuals} or --all")

    logger.info("Cache manifest: %s", _manifest_path())


if __name__ == "__main__":
    main()
