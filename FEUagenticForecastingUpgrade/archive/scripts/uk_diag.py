#!/usr/bin/env python3
"""
UK diagnosis — where does the OLD model (`predicted`) lose to the LEGO benchmark
(`prediction_rf`), and what is the bar to beat in every category?

Two questions:
  1. Head-to-head (the 2 categories the old model covers — DEODORANT & FRAGRANCE,
     HOME & HYGIENE): predicted vs prediction_rf at lag 4 & 5, sliced by lego_segment,
     demand pattern, and volume tier — quantify WHERE and the bias direction.
  2. The bar (all 14 in-scope categories): prediction_rf's own accuracy & |bias| vs
     actuals per category/lag — this is what the new model must beat.

Inputs: the local cache (datacache/uk/benchmark + actuals) via utils.uk_eval.
Outputs: CSVs under notebooks/uk_diag_outputs/ + a printed summary.

Run: .venv/bin/python notebooks/uk_diag.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import uk_eval as E  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("uk_diag")

OUT = Path("notebooks/uk_diag_outputs")
PREDICTED_CATS = ["DEODORANT & FRAGRANCE", "HOME & HYGIENE"]


# --------------------------------------------------------------------------- #
# demand pattern (Syntetos–Boylan) + volume tier, from the actuals series
# --------------------------------------------------------------------------- #
def classify_keys(keys: pd.Index) -> pd.DataFrame:
    """Per-key demand pattern + total volume, computed from the cached actuals
    history (weeks <= 202619). Returns [key, adi, cv2, pattern, total, vol_tier]."""
    acts = E._read_actuals()
    acts = acts[acts["key"].isin(set(keys))]
    out = []
    for key, g in acts.groupby("key"):
        v = g["actual"].values.astype("float64")
        n = len(v)
        nz = v[v > 0]
        total = float(v.sum())
        if n == 0 or len(nz) == 0:
            out.append((key, np.nan, np.nan, "dead", total))
            continue
        adi = n / len(nz)
        cv2 = (nz.std() / nz.mean()) ** 2 if nz.mean() > 0 else np.nan
        if adi < 1.32 and (cv2 is np.nan or cv2 < 0.49):
            pat = "smooth"
        elif adi >= 1.32 and cv2 < 0.49:
            pat = "intermittent"
        elif adi < 1.32 and cv2 >= 0.49:
            pat = "erratic"
        else:
            pat = "lumpy"
        out.append((key, adi, cv2, pat, total))
    df = pd.DataFrame(out, columns=["key", "adi", "cv2", "pattern", "total"])
    df["vol_tier"] = pd.qcut(df["total"].rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_top"])
    return df


def _bar_table(frame: pd.DataFrame, pred_col: str, label: str) -> pd.DataFrame:
    """Accuracy & bias of a single forecast column vs actual, per (category, lag)."""
    recs = []
    for (cat, h), g in frame.groupby(["category_name", "horizon"]):
        acc, bias, sa = E.acc_bias(g["actual"].values, g[pred_col].values)
        recs.append({"category": cat, "lag": int(h), "n": len(g), "sum_actual": sa,
                     f"acc_{label}": acc, f"bias_{label}": bias})
    for cat, g in frame.groupby("category_name"):
        acc, bias, sa = E.acc_bias(g["actual"].values, g[pred_col].values)
        recs.append({"category": cat, "lag": "4&5", "n": len(g), "sum_actual": sa,
                     f"acc_{label}": acc, f"bias_{label}": bias})
    return pd.DataFrame(recs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snaps = E.DEFAULT_SNAPSHOTS
    fr = E.load_scoring_frame(snaps, lags=(4, 5))

    # 0) Harness self-check: prediction_rf vs itself must show zero gap, no win.
    chk = E.scoreboard(fr, cand_col="prediction_rf", missing_cand="zero")
    allrow = chk[(chk.category == "__ALL__") & (chk.lag == "4&5")].iloc[0]
    logger.info("self-check rf-vs-rf  acc_gap=%.4g  WIN=%s (expect ~0, False)", allrow["acc_gap"], allrow["WIN"])

    # 1) Head-to-head: predicted vs prediction_rf on the 2 covered categories.
    sb_pred = E.scoreboard(fr, cand_col="predicted", missing_cand="drop")
    sb_pred.to_csv(OUT / "predicted_vs_rf.csv", index=False)
    print("\n=== predicted vs prediction_rf (2 covered categories, lag 4 & 5) ===")
    show = sb_pred[sb_pred.category.isin(PREDICTED_CATS)][
        ["category", "lag", "n", "acc_cand", "acc_lego", "acc_gap", "bias_cand", "bias_lego", "WIN"]]
    print(show.to_string(index=False))

    # 2) The bar: prediction_rf accuracy & |bias| per category (all 14).
    bar = _bar_table(fr, "prediction_rf", "rf")
    bar = bar.sort_values(["category", "lag"]).reset_index(drop=True)
    bar.to_csv(OUT / "rf_bar_by_category.csv", index=False)
    print("\n=== prediction_rf — the bar to beat (lag 4&5 rows) ===")
    print(bar[bar.lag == "4&5"][["category", "n", "sum_actual", "acc_rf", "bias_rf"]].to_string(index=False))

    # 3) Demand pattern + volume tier slices.
    cls = classify_keys(pd.Index(fr["key"].unique()))
    cls.to_csv(OUT / "key_classification.csv", index=False)
    frc = fr.merge(cls[["key", "pattern", "vol_tier"]], on="key", how="left")

    # 3a) rf accuracy/bias by (category, pattern) — all cats (where does the bar sag?)
    rec = []
    for (cat, pat), g in frc.groupby(["category_name", "pattern"]):
        acc, bias, sa = E.acc_bias(g["actual"].values, g["prediction_rf"].values)
        rec.append({"category": cat, "pattern": pat, "n": len(g), "sum_actual": sa, "acc_rf": acc, "bias_rf": bias})
    rf_by_pat = pd.DataFrame(rec).sort_values(["category", "pattern"])
    rf_by_pat.to_csv(OUT / "rf_by_pattern.csv", index=False)

    # 3b) predicted vs rf by (category, lego_segment) for the 2 covered cats.
    cov = frc[frc.category_name.isin(PREDICTED_CATS) & frc["predicted"].notna()]
    rec = []
    for (cat, seg), g in cov.groupby(["category_name", "lego_segment"]):
        acc_p, bias_p, sa = E.acc_bias(g["actual"].values, g["predicted"].values)
        acc_r, bias_r, _ = E.acc_bias(g["actual"].values, g["prediction_rf"].values)
        rec.append({"category": cat, "lego_segment": seg, "n": len(g), "sum_actual": sa,
                    "acc_pred": acc_p, "acc_rf": acc_r, "acc_gap": acc_p - acc_r,
                    "bias_pred": bias_p, "bias_rf": bias_r})
    seg_tbl = pd.DataFrame(rec).sort_values(["category", "sum_actual"], ascending=[True, False])
    seg_tbl.to_csv(OUT / "predicted_vs_rf_by_segment.csv", index=False)

    # 4) Headline.
    print("\n=== headline ===")
    for cat in PREDICTED_CATS:
        r = sb_pred[(sb_pred.category == cat) & (sb_pred.lag == "4&5")]
        if len(r):
            r = r.iloc[0]
            verdict = "WIN" if r["WIN"] else ("acc only" if r["acc_win"] else ("bias only" if r["bias_win"] else "LOSE"))
            print(f"  {cat:24s} lag4&5: acc {r['acc_cand']:.3f} vs rf {r['acc_lego']:.3f} | "
                  f"bias {r['bias_cand']:+.3f} vs rf {r['bias_lego']:+.3f} -> {verdict}")
    print(f"\nOutputs written to {OUT}/")


if __name__ == "__main__":
    main()
