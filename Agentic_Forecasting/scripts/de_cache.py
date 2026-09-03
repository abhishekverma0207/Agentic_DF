#!/usr/bin/env python3
"""
Build a LOCAL parquet cache of the DE (Germany) parallel_run data for local-first
segment-model research.

Sources (per snapshot under
  /Volumes/pds_feu_931272_dev/eu_dach/transform/DACH/DACH_LEGO/Germany/parallel_run/{snap}/All_Cat/):
  - DT/                  the modelling panel (224 cols; target=Actuals, has lego_segment)
  - FS/                  the feature store (177 f_* Fourier / promo lead-lag features)
  - TF_pristine_allkeys/ LEGO benchmark (prediction_xgb) + target + lego_segment

Layout written:
  datacache/de/parallel_run/
    dt/dt_{snap}.parquet           curated DT (ids/target/attrs/signals/promo/holiday/season + lego_segment)
    fs/fs_{snap}.parquet           key,year_week + f_*
    benchmark/bench_{snap}.parquet key,year_week,prediction_xgb,category,lego_segment,target,cs_gtin
    actuals/actuals_{ASNAP}.parquet  realized forward actuals (key,year_week,Actuals)
    manifest.json

The DT compresses to ~200 MB/snapshot, so each dir is read whole (column-projected,
uc_io parallel across parts), row-filtered to the in-scope categories + the
[HISTORY_FLOOR .. snap+HORIZONS] week window, and written as one snappy parquet.

CLI:
    .venv/bin/python scripts/de_cache.py --all
    .venv/bin/python scripts/de_cache.py --kind panel --snapshots 202601,202606
    .venv/bin/python scripts/de_cache.py --kind benchmark --snapshots 202601..202621
    .venv/bin/python scripts/de_cache.py --kind actuals
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import uc_io                       # noqa: E402
from utils.uk_forecast import add_weeks       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("de_cache")

BASE = "/Volumes/pds_feu_931272_dev/eu_dach/transform/DACH/DACH_LEGO/Germany/parallel_run"
CACHE = Path("datacache/de/parallel_run")

IN_SCOPE = {"SCRATCH COOKING AIDS", "DEODORANTS & FRAGRANCES", "DRESSINGS", "SKIN CLEANSING",
            "HEALTHY SNACKING", "HOME & HYGIENE", "FABRIC CLEANING", "ORAL CARE", "SKIN CARE", "HAIR CARE"}

TRAIN_SNAPSHOTS = ["202601", "202606", "202611", "202616", "202621"]
ACTUALS_SNAPSHOT = "202625"
HORIZONS = 13
HISTORY_FLOOR = "202401"   # ~2 years of history before the earliest train_end

# --- DT column selection (built dynamically from the schema) ---
DT_IDS = ["key", "year_week", "Category", "GTIN", "Actuals", "lego_segment", "exclude_forecast",
          "Brand", "APG", "Segment", "ForecastFamilyDescription", "smooth_sales", "smooth_sales_orig"]
DT_SIGNALS = ["POS_actual_sales", "SS_actual_sales", "actual_dispatched_quantity",
              "expected_dispatched_quantity", "Dispatch_rate", "pricing_list_price",
              "PromoFlag", "Promo_Week", "PromoStart", "PromoSecond", "Duration", "PromoID",
              "HolidayFlag", "total_holidays"]
DT_SEASON = ["summer", "winter", "spring", "autumn", "high_sunshine", "low_sunshine",
             "month_end", "month_start", "quarter_end", "quarter_start", "covid"]
DT_PROMO_PREFIX = ("PromoMechanic_", "PromoMechanic2_", "PromoFeature_", "PromoCommunication_")
DT_HOLIDAY_PREFIX = ("holiday_",)
# Dropped: raw weather (leakage), index/removal flags, free-text remarks/descriptions.
DT_DROP_EXACT = {"avgTemp", "maxTemp", "minTemp", "prcp", "sunHours"}


def _dt_keep(schema_cols: List[str]) -> List[str]:
    s = set(schema_cols)
    keep = [c for c in DT_IDS + DT_SIGNALS + DT_SEASON if c in s]
    for c in schema_cols:
        if c in keep or c in DT_DROP_EXACT:
            continue
        if c.startswith(DT_PROMO_PREFIX) or c.startswith(DT_HOLIDAY_PREFIX):
            keep.append(c)
    return list(dict.fromkeys(keep))


def _yw(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("-", "", regex=False)


def _manifest(entry: str, info: dict) -> None:
    mp = CACHE / "manifest.json"
    data = {}
    if mp.exists():
        try:
            data = json.loads(mp.read_text())
        except Exception:
            data = {}
    data[entry] = info
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(data, indent=2, sort_keys=True))


def cache_panel(snap: str, *, overwrite: bool = False) -> None:
    hi = add_weeks(snap, HORIZONS)
    for kind, sub, builder in [("dt", "DT", _dt_panel), ("fs", "FS", _fs_panel)]:
        dst = CACHE / kind / f"{kind}_{snap}.parquet"
        if dst.exists() and not overwrite:
            logger.info("[%s] %s cached -> %s", kind, snap, dst); continue
        df = builder(snap, hi)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        _manifest(f"{kind}:{snap}", {"rows": int(len(df)), "cols": int(df.shape[1]),
                  "bytes": dst.stat().st_size, "window": [HISTORY_FLOOR, hi],
                  "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        logger.info("[%s] %s done: %s rows x %d cols, %.1f MB", kind, snap, f"{len(df):,}",
                    df.shape[1], dst.stat().st_size / 1e6)


def _dt_panel(snap: str, hi: str) -> pd.DataFrame:
    schema = uc_io.read_parquet(f"{BASE}/{snap}/All_Cat/DT", max_parts=1).columns.tolist()
    keep = _dt_keep(schema)
    logger.info("[dt] %s reading %d/%d cols", snap, len(keep), len(schema))
    d = uc_io.read_parquet(f"{BASE}/{snap}/All_Cat/DT", columns=keep, max_workers=16)
    d["key"] = d["key"].astype(str)
    yw = _yw(d["year_week"])
    d = d[d["Category"].isin(IN_SCOPE) & (yw >= HISTORY_FLOOR) & (yw <= hi)].copy()
    d["cs_gtin"] = d["key"].str.rsplit("_", n=1).str[0]
    return d


def _fs_panel(snap: str, hi: str) -> pd.DataFrame:
    schema = uc_io.read_parquet(f"{BASE}/{snap}/All_Cat/FS", max_parts=1).columns.tolist()
    keep = [c for c in ["key", "year_week"] if c in schema] + [c for c in schema if c.startswith("f_")]
    logger.info("[fs] %s reading %d f_* (+ids) of %d", snap, len(keep) - 2, len(schema))
    d = uc_io.read_parquet(f"{BASE}/{snap}/All_Cat/FS", columns=keep, max_workers=16)
    d["key"] = d["key"].astype(str)
    yw = _yw(d["year_week"])
    return d[(yw >= HISTORY_FLOOR) & (yw <= hi)].copy()


def cache_benchmark(snap: str, *, overwrite: bool = False) -> None:
    dst = CACHE / "benchmark" / f"bench_{snap}.parquet"
    if dst.exists() and not overwrite:
        logger.info("[benchmark] %s cached", snap); return
    cols = ["key", "year_week", "prediction_xgb", "category", "lego_segment", "target", "cs_gtin",
            "predicted", "ml_pred_final", "Period_Type"]
    schema = uc_io.read_parquet(f"{BASE}/{snap}/All_Cat/TF_pristine_allkeys", max_parts=1).columns.tolist()
    d = uc_io.read_parquet(f"{BASE}/{snap}/All_Cat/TF_pristine_allkeys",
                           columns=[c for c in cols if c in schema], max_workers=16)
    d["key"] = d["key"].astype(str)
    if "category" in d.columns:
        d = d[d["category"].isin(IN_SCOPE)].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(dst, index=False)
    _manifest(f"benchmark:{snap}", {"rows": int(len(d)), "bytes": dst.stat().st_size,
              "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    logger.info("[benchmark] %s done: %s rows, %.1f MB", snap, f"{len(d):,}", dst.stat().st_size / 1e6)


def cache_actuals(*, overwrite: bool = False) -> None:
    dst = CACHE / "actuals" / f"actuals_{ACTUALS_SNAPSHOT}.parquet"
    if dst.exists() and not overwrite:
        logger.info("[actuals] cached"); return
    d = uc_io.read_parquet(f"{BASE}/{ACTUALS_SNAPSHOT}/All_Cat/DT",
                           columns=["key", "year_week", "Actuals", "Category"], max_workers=16)
    d["key"] = d["key"].astype(str)
    d = d[d["Category"].isin(IN_SCOPE)][["key", "year_week", "Actuals"]].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(dst, index=False)
    _manifest("actuals", {"rows": int(len(d)), "snapshot": ACTUALS_SNAPSHOT, "bytes": dst.stat().st_size,
              "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    logger.info("[actuals] done: %s rows, %.1f MB", f"{len(d):,}", dst.stat().st_size / 1e6)


def _expand(spec: str) -> List[str]:
    spec = spec.strip()
    if ".." in spec:
        a, b = (x.strip().replace("-", "") for x in spec.split("..", 1))
        out, cur = [], a
        for _ in range(60):
            out.append(cur)
            if cur == b:
                break
            cur = add_weeks(cur, 5)   # snapshots are 5 weeks apart
        return out
    return [x.strip().replace("-", "") for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache DE parallel_run data -> datacache/de/parallel_run/")
    ap.add_argument("--kind", choices=["panel", "benchmark", "actuals"])
    ap.add_argument("--all", action="store_true", help="panel+benchmark for TRAIN_SNAPSHOTS + actuals")
    ap.add_argument("--snapshots", default="")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    snaps = _expand(a.snapshots) if a.snapshots else TRAIN_SNAPSHOTS
    if a.all:
        for s in snaps:
            cache_panel(s, overwrite=a.overwrite)
            cache_benchmark(s, overwrite=a.overwrite)
        cache_actuals(overwrite=a.overwrite)
    elif a.kind == "panel":
        for s in snaps:
            cache_panel(s, overwrite=a.overwrite)
    elif a.kind == "benchmark":
        for s in snaps:
            cache_benchmark(s, overwrite=a.overwrite)
    elif a.kind == "actuals":
        cache_actuals(overwrite=a.overwrite)
    else:
        ap.error("pass --all or --kind {panel,benchmark,actuals}")
    logger.info("manifest: %s", CACHE / "manifest.json")


if __name__ == "__main__":
    main()
