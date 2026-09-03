#!/usr/bin/env python3
"""
Phase-4 accuracy screen: the 6 categories that trail prediction_rf on accuracy at
GTIN lag-4 (CONDIMENT, SKIN CLEANSING, FABRIC CLEANING, FABRIC ENHANCER, SKIN CARE,
HAIR CARE) need genuine predictive gains — calibration can't fix accuracy.

This screens GBM CAPACITY variants (more trees / lower LR / deeper) at FULL strength
(no sub-sampling — capacity only shows with full data) on a 4-origin spread, ranked
on GTIN lag-4 accuracy of the laggards (and overall wins). The winner then gets a
full 7-origin run via run_uk_local.py.

Run: .venv/bin/python scripts/uk_accuracy_experiment.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import uk_eval as E       # noqa: E402
from utils import uk_forecast as F   # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SNAPS = ["202609", "202611", "202613", "202615"]
OUT = Path("artifacts_uk_local/experiments_acc")
LAGGARDS = ["CONDIMENT", "SKIN CLEANSING", "FABRIC CLEANING", "FABRIC ENHANCER", "SKIN CARE", "HAIR CARE"]

# Calibration OFF so the screen isolates the per-category effect on accuracy.
BASE_FULL = {"train_sample_frac": 1.0, "bias_calibration": False}
VARIANTS = [
    ("global",            {"per_category": False}),                       # current pooled model
    ("percat_all",        {"per_category": True}),                        # one GBM per category
    ("percat_laggards",   {"per_category": True, "per_category_categories": LAGGARDS}),  # per-cat for laggards only
]


def run_variant(name, overrides, fr):
    cfg = F.load_config(overrides={**BASE_FULL, **overrides})
    t0 = time.time()
    fcs = [F.forecast_origin(s, cfg, horizons=[4, 5]).assign(snapshot=s) for s in SNAPS]
    allfc = pd.concat(fcs, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    allfc.to_parquet(OUT / f"forecasts_{name}.parquet", index=False)
    fr2 = E.attach_candidate(fr, allfc, value_col="forecast", name="c")
    sb = E.scoreboard(fr2, cand_col="c", missing_cand="zero", level="gtin")
    sb.to_csv(OUT / f"scoreboard_{name}_gtin.csv", index=False)
    l4 = sb[(sb.category != "__ALL__") & (sb.lag == 4)].set_index("category")
    allr = sb[(sb.category == "__ALL__") & (sb.lag == 4)].iloc[0]
    lag_acc = {c: round(float(l4.loc[c, "acc_cand"]), 3) for c in LAGGARDS if c in l4.index}
    lag_accwin = sum(int(l4.loc[c, "acc_win"]) for c in LAGGARDS if c in l4.index)
    return {"variant": name, "secs": round(time.time() - t0),
            "wins_l4": int(l4["WIN"].sum()), "acc_l4": round(allr["acc_cand"], 4),
            "accrf_l4": round(allr["acc_lego"], 4), "laggard_accwins": lag_accwin,
            **{f"acc_{c[:6]}": v for c, v in lag_acc.items()}}


def main():
    print(f"Accuracy screen (FULL strength), {len(SNAPS)} origins, GTIN lag-4. "
          f"rf laggard accs to beat:\n")
    fr = E.load_scoring_frame(SNAPS, lags=(4, 5))
    # show the bar (rf acc) for laggards once
    sb0 = E.scoreboard(E.attach_candidate(fr, fr.assign(f=fr["prediction_rf"]), value_col="f", name="c"),
                       cand_col="c", missing_cand="zero", level="gtin")
    l4 = sb0[(sb0.lag == 4)].set_index("category")
    print("  rf acc:", {c[:6]: round(float(l4.loc[c, "acc_lego"]), 3) for c in LAGGARDS if c in l4.index}, "\n")
    rows = [run_variant(n, ov, fr) for n, ov in VARIANTS]
    for r in rows:
        print(f"  {r['variant']:12s} wins_l4={r['wins_l4']:2d} acc={r['acc_l4']:.3f}(rf {r['accrf_l4']:.3f}) "
              f"laggard_accwins={r['laggard_accwins']}/6 [{r['secs']}s]")
    comp = pd.DataFrame(rows).sort_values(["wins_l4", "laggard_accwins", "acc_l4"], ascending=False)
    comp.to_csv(OUT / "accuracy_comparison.csv", index=False)
    print("\n=== ranked ===")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
