#!/usr/bin/env python3
"""
FINALIZE: full 7-origin hierarchical stacked ensemble.

  base models : GLOBAL (pooled) + LOCAL (per-category), full strength, calibration off.
  stacker     : per-category convex weight (local share), learned LEAVE-ONE-ORIGIN-OUT
                (weights from the other 6 origins; applied to the held-out one — leakage-safe).
  + calib     : DEODORANT-only damped bias factor, also learned leave-one-origin-out.

Reuses the 4 saved screen origins (experiments_acc) and generates the 3 missing ones.
Reports the definitive GTIN lag-4/5 result (primary) + key-level, and saves the scoreboard.

Run: .venv/bin/python scripts/uk_stack_final.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import uk_eval as E       # noqa: E402
from utils import uk_forecast as F   # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SN = ["202609", "202610", "202611", "202612", "202613", "202614", "202615"]
STACK = Path("artifacts_uk_local/stack_final")
DEO = "DEODORANT & FRAGRANCE"


def base_forecasts():
    """(global_df, percat_df) for all 7 origins — reuse 4 saved screen origins, generate the rest."""
    gp, lp = STACK / "global_7.parquet", STACK / "percat_7.parquet"
    if gp.exists() and lp.exists():
        return pd.read_parquet(gp), pd.read_parquet(lp)
    STACK.mkdir(parents=True, exist_ok=True)
    gparts, lparts, have = [], [], set()
    for f, parts in [("forecasts_global.parquet", gparts), ("forecasts_percat_all.parquet", lparts)]:
        p = Path("artifacts_uk_local/experiments_acc") / f
        if p.exists():
            d = pd.read_parquet(p)
            d["snapshot"] = d["snapshot"].astype(str)
            parts.append(d)
            have = set(d["snapshot"].unique())
    gcfg = F.load_config(overrides={"per_category": False, "bias_calibration": False})
    lcfg = F.load_config(overrides={"per_category": True, "bias_calibration": False})
    for s in SN:
        if s in have:
            continue
        print(f"  generating base forecasts for {s} (missing)…", flush=True)
        gparts.append(F.forecast_origin(s, gcfg, horizons=[4, 5]).assign(snapshot=s))
        lparts.append(F.forecast_origin(s, lcfg, horizons=[4, 5]).assign(snapshot=s))
    g = pd.concat(gparts, ignore_index=True)
    l = pd.concat(lparts, ignore_index=True)
    g = g[g["snapshot"].isin(SN)]; l = l[l["snapshot"].isin(SN)]
    g.to_parquet(gp, index=False); l.to_parquet(lp, index=False)
    return g, l


def main():
    g, l = base_forecasts()
    fr = E.load_scoring_frame(SN, lags=(4, 5))
    fr = E.attach_candidate(fr, g, value_col="forecast", name="g")
    fr = E.attach_candidate(fr, l, value_col="forecast", name="l")
    fr["g"] = fr["g"].fillna(0.0); fr["l"] = fr["l"].fillna(0.0)

    def blend(frame, wmap):
        wv = frame["category_name"].map(wmap).fillna(0.0)
        return wv * frame["l"] + (1.0 - wv) * frame["g"]

    parts = []
    for test in SN:
        tr = fr[fr.snapshot != test]
        trg = tr.groupby(["category_name", "cs_gtin", "year_week"], as_index=False).agg(
            g=("g", "sum"), l=("l", "sum"), actual=("actual", "sum"))
        wmap = F._learn_blend_weights(trg, "category_name")
        te = fr[fr.snapshot == test].copy()
        te["stacked"] = blend(te, wmap)
        # DEODORANT-only calibration, learned leave-one-out on the blended train forecast (GTIN level)
        trb = tr.copy(); trb["stacked"] = blend(trb, wmap)
        deo = trb[trb.category_name == DEO]
        dg = deo.groupby(["cs_gtin", "year_week"]).agg(a=("actual", "sum"), s=("stacked", "sum"))
        if dg["s"].sum() > 0:
            fac = float(np.clip(1 + 0.7 * (dg["a"].sum() / dg["s"].sum() - 1), 0.85, 1.15))
            te.loc[te.category_name == DEO, "stacked"] *= fac
        parts.append(te)
    S = pd.concat(parts, ignore_index=True)
    S["stacked"] = S["stacked"].clip(lower=0)

    sb = E.scoreboard(S, cand_col="stacked", missing_cand="zero", level="gtin")
    sb.to_csv(STACK / "scoreboard_stacked_gtin.csv", index=False)
    print("\n================  FINAL: 7-origin stacked + DEODORANT calib  ================")
    for lg in (4, 5):
        w = sb[(sb.category != "__ALL__") & (sb.lag == lg)]
        a = sb[(sb.category == "__ALL__") & (sb.lag == lg)].iloc[0]
        print(f"GTIN lag{lg}: WINS {int(w.WIN.sum())}/13 | acc {a['acc_cand']:.3f} vs rf {a['acc_lego']:.3f} | "
              f"bias {a['bias_cand']:+.3f} vs rf {a['bias_lego']:+.3f}")
    sk = E.scoreboard(S, cand_col="stacked", missing_cand="zero", level="key")
    for lg in (4, 5):
        a = sk[(sk.category == "__ALL__") & (sk.lag == lg)].iloc[0]
        print(f"KEY  lag{lg}: acc {a['acc_cand']:.3f} vs rf {a['acc_lego']:.3f}")
    print("\nGTIN lag-4 per category:")
    d = sb[(sb.category != "__ALL__") & (sb.lag == 4)][
        ["category", "n", "acc_cand", "acc_lego", "bias_cand", "bias_lego", "WIN"]]
    print(d.sort_values(["WIN", "acc_cand"], ascending=False).to_string(index=False))
    print(f"\nWinners (GTIN lag4): {sorted(d[d.WIN].category.tolist())}")
    print(f"Saved -> {STACK}/scoreboard_stacked_gtin.csv")


if __name__ == "__main__":
    main()
