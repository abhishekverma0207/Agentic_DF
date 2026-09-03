#!/usr/bin/env python3
"""
Phase-4 lever screen: run config variants on rolling origins and rank them on the
metric that matters most — categories WON (acc + |bias|) at **lag 4, GTIN-level**.

Each lag is scored independently (no pooling). Reports BOTH levels:
  * gtin (PRIMARY) — forecast & actual summed to cs_gtin × year_week before error.
  * key            — raw gtin×customer × year_week grain.
And both lags (4 primary, 5 secondary).

v1 (tweedie) wins bias broadly but loses WAPE accuracy by small margins. WAPE is
L1-like, so we screen objectives that align with it (quantile@~0.5, L1). Per-variant
forecasts are saved so we can re-score at any level/lag without retraining.

Run: .venv/bin/python scripts/uk_experiment.py            # fast screen, 4 origins
     .venv/bin/python scripts/uk_experiment.py --full      # full-strength
"""
from __future__ import annotations

import argparse
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
SNAPS = ["202609", "202611", "202613", "202615"]   # 4-origin screen spread
OUT = Path("artifacts_uk_local/experiments")

# Gentle calibration bounds — wide [0.5,1.5] over-corrected to bias -0.13 in testing.
_CALIB = {"bias_calibration": True, "bias_factor_min": 0.8, "bias_factor_max": 1.2}
VARIANTS = [
    ("tweedie_v13",        {"objective": "tweedie", "tweedie_variance_power": 1.3}),       # v1 baseline
    ("quantile_055",       {"objective": "quantile", "alpha": 0.55}),                      # WAPE-aligned
    ("quantile_060",       {"objective": "quantile", "alpha": 0.60}),                      # protect volume
    ("tweedie_calib",      {"objective": "tweedie", "tweedie_variance_power": 1.3, **_CALIB}),
    ("quantile_055_calib", {"objective": "quantile", "alpha": 0.55, **_CALIB}),
]


def run_variant(name, overrides, mode_over, fr):
    cfg = F.load_config(overrides={**mode_over, **overrides})
    t0 = time.time()
    fcs = [F.forecast_origin(s, cfg, horizons=[4, 5]).assign(snapshot=s) for s in SNAPS]
    allfc = pd.concat(fcs, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    allfc.to_parquet(OUT / f"forecasts_{name}.parquet", index=False)
    fr2 = E.attach_candidate(fr, allfc, value_col="forecast", name="c")
    res = {"variant": name, "secs": round(time.time() - t0), "coverage": round(fr2["c"].notna().mean(), 3)}
    for level in ("gtin", "key"):
        sb = E.scoreboard(fr2, cand_col="c", missing_cand="zero", level=level)
        sb.to_csv(OUT / f"scoreboard_{name}_{level}.csv", index=False)
        for lg in (4, 5):
            wins = sb[(sb.category != "__ALL__") & (sb.lag == lg)]
            allr = sb[(sb.category == "__ALL__") & (sb.lag == lg)].iloc[0]
            res[f"wins_{level}_l{lg}"] = int(wins.WIN.sum())
            res[f"acc_{level}_l{lg}"] = round(allr["acc_cand"], 4)
            res[f"accrf_{level}_l{lg}"] = round(allr["acc_lego"], 4)
            res[f"bias_{level}_l{lg}"] = round(allr["bias_cand"], 4)
            res[f"biasrf_{level}_l{lg}"] = round(allr["bias_lego"], 4)
            if level == "gtin" and lg == 4:
                res["won_gtin_l4"] = ", ".join(c[:8] for c in sorted(wins[wins.WIN].category))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    mode_over = {} if a.full else {"train_sample_frac": 0.5, "n_estimators": 400}
    print(f"Screening {len(VARIANTS)} variants, mode={'full' if a.full else 'fast'}, "
          f"{len(SNAPS)} origins — ranking on GTIN lag-4 wins (each lag scored independently)\n")
    fr = E.load_scoring_frame(SNAPS, lags=(4, 5))
    rows = []
    for name, ov in VARIANTS:
        r = run_variant(name, ov, mode_over, fr)
        rows.append(r)
        print(f"  {r['variant']:14s} GTIN l4: wins={r['wins_gtin_l4']:2d}/13 acc={r['acc_gtin_l4']:.3f}"
              f"(rf {r['accrf_gtin_l4']:.3f}) bias={r['bias_gtin_l4']:+.3f}(rf {r['biasrf_gtin_l4']:+.3f}) | "
              f"GTIN l5 wins={r['wins_gtin_l5']:2d} | KEY l4 wins={r['wins_key_l4']:2d} [{r['secs']}s]")
    comp = pd.DataFrame(rows).sort_values(["wins_gtin_l4", "acc_gtin_l4"], ascending=False)
    comp.to_csv(OUT / "variant_comparison.csv", index=False)
    print("\n=== ranked by GTIN lag-4 wins ===")
    cols = ["variant", "wins_gtin_l4", "acc_gtin_l4", "accrf_gtin_l4", "bias_gtin_l4", "biasrf_gtin_l4",
            "wins_gtin_l5", "wins_key_l4", "won_gtin_l4"]
    print(comp[cols].to_string(index=False))
    print(f"\nSaved -> {OUT}/variant_comparison.csv + per-variant forecasts & scoreboards (gtin/key)")


if __name__ == "__main__":
    main()
