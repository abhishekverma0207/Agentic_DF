#!/usr/bin/env python3
"""
Local UK rolling-origin runner — train the deterministic UK model per origin,
forecast the full non-excluded universe, and score vs the LEGO `prediction_rf`
benchmark at lag 4 & 5 (pooled across origins, per category).

Mirrors run_th_local.py in spirit. Reads the local datacache (no Databricks),
so iteration is fast; the model logic in utils/uk_forecast.py migrates to the
Databricks job later.

Examples:
  # full-strength, all 7 origins, lag 4 & 5 (the headline eval)
  .venv/bin/python run_uk_local.py --snapshots 2026-09..2026-15 --horizons 4,5 --mode full
  # fast smoke (half-sample, fewer trees)
  .venv/bin/python run_uk_local.py --snapshots 2026-09 --horizons 4,5 --mode fast
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import uk_eval as E          # noqa: E402
from utils import uk_forecast as F      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_uk_local")

MODE_OVERRIDES = {
    "fast": {"train_sample_frac": 0.5, "n_estimators": 400},
    "full": {},
}


def _expand(spec: str):
    spec = spec.strip()
    if ".." in spec:
        a, b = (x.replace("-", "") for x in spec.split("..", 1))
        out, cur = [], a
        for _ in range(60):
            out.append(cur)
            if cur == b:
                break
            cur = F.add_weeks(cur, 1)
        return out
    return [x.replace("-", "") for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="2026-09..2026-15")
    ap.add_argument("--horizons", default="4,5", help="comma list, e.g. '4,5' or '1,2,3,4,5,6,7,8,9,10,11,12,13'")
    ap.add_argument("--mode", choices=["fast", "full"], default="full")
    ap.add_argument("--config", default=F.BASE_CONFIG)
    ap.add_argument("--out-dir", default="artifacts_uk_local/run")
    ap.add_argument("--label", default="uk")
    ap.add_argument("--overrides", default="", help="JSON dict merged into the config, e.g. "
                    "'{\"bias_calibration\": true, \"calibrate_categories\": [\"HOME & HYGIENE\"]}'")
    a = ap.parse_args()

    snaps = _expand(a.snapshots)
    horizons = [int(x) for x in a.horizons.split(",") if x.strip()]
    overrides = dict(MODE_OVERRIDES[a.mode])
    if a.overrides.strip():
        import json
        overrides.update(json.loads(a.overrides))
    cfg = F.load_config(a.config, overrides=overrides)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("snapshots=%s horizons=%s mode=%s", snaps, horizons, a.mode)

    forecasts, timings = [], {}
    for s in snaps:
        t0 = time.time()
        try:
            fc = F.forecast_origin(s, cfg, horizons=horizons)
        except FileNotFoundError as exc:
            logger.warning("  skip %s — %s", s, exc)
            continue
        if fc.empty:
            logger.warning("  %s produced no forecasts", s)
            continue
        fc["snapshot"] = s
        forecasts.append(fc)
        timings[s] = time.time() - t0
        logger.info("  %s: %s forecast rows in %.0fs", s, f"{len(fc):,}", timings[s])

    if not forecasts:
        logger.error("No forecasts produced — aborting.")
        return
    allfc = pd.concat(forecasts, ignore_index=True)
    allfc.to_parquet(out / "uk_forecasts.parquet", index=False)

    # Score across origins vs prediction_rf — each lag independently, both levels.
    fr = E.load_scoring_frame(snaps, lags=horizons)
    fr2 = E.attach_candidate(fr, allfc, value_col="forecast", name=a.label)
    coverage = fr2[a.label].notna().mean()
    for level in ("gtin", "key"):                       # gtin = PRIMARY
        sb = E.scoreboard(fr2, cand_col=a.label, missing_cand="zero", level=level)
        sb.to_csv(out / f"uk_scoreboard_{level}.csv", index=False)
        print(f"\n=== {a.label} vs prediction_rf — {level.upper()} level, {snaps[0]}..{snaps[-1]} "
              f"(coverage {coverage:.3f}) ===")
        mat = sb[sb.category != "__ALL__"][["category", "lag", "n", "acc_cand", "acc_lego",
                                            "bias_cand", "bias_lego", "WIN"]]
        with pd.option_context("display.width", 220, "display.max_rows", None):
            print(mat.to_string(index=False))
        for lg in horizons:
            wins = sb[(sb.category != "__ALL__") & (sb.lag == lg)]
            allr = sb[(sb.category == "__ALL__") & (sb.lag == lg)]
            if len(wins) and len(allr):
                allr = allr.iloc[0]
                print(f"  lag {lg}: cats won {int(wins['WIN'].sum())}/{len(wins)} | "
                      f"__ALL__ acc {allr['acc_cand']:.3f} vs rf {allr['acc_lego']:.3f} | "
                      f"bias {allr['bias_cand']:+.3f} vs rf {allr['bias_lego']:+.3f}")
    print(f"\nForecasts -> {out}/uk_forecasts.parquet | scoreboards -> {out}/uk_scoreboard_{{gtin,key}}.csv")
    print(f"timings(s): { {k: round(v) for k, v in timings.items()} }")


if __name__ == "__main__":
    main()
