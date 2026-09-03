#!/usr/bin/env python
"""
HEAD-aware blend — test whether splitting the blend grain by volume TIER (head/tail)
lets the category model deploy on the head (where it beats LEGO) instead of being
diluted by the tail. Operates on CACHED components (instant).

Compares blend grains out-of-sample (fit on N-1 snapshots, eval held-out):
  - category|segment           (current production-candidate)
  - category|tier              (head/tail only)
  - category|segment|tier      (both)

    .venv/bin/python scripts/de_head_blend.py --nboost 700
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.weight_fit import fit_per_category_weights   # noqa: E402

CACHE = Path("datacache/de/parallel_run")
WEEK = "year_week"
MIN_VOL = 1000.0
SHRINK_K = 5000.0


def _acc(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return 1.0 - np.abs(f - a).sum() / max(a.sum(), 1e-9)


def _bias(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return (f - a).sum() / max(a.sum(), 1e-9)


def load(snaps, nboost, lags):
    acts = pd.read_parquet(CACHE / "actuals" / "actuals_202625.parquet")
    acts["key"] = acts["key"].astype(str)
    acts[WEEK] = acts[WEEK].astype(str).str.replace("-", "", regex=False)
    acts = acts.rename(columns={"Actuals": "actual"}).groupby(["key", WEEK], as_index=False)["actual"].sum()
    out = {}
    for s in snaps:
        c = pd.read_parquet(CACHE / "components" / f"comp_{s}_h4-5_nb{nboost}.parquet")
        out[s] = c[c["horizon"].isin(lags)].merge(acts, on=["key", WEEK], how="inner")
    return out


def add_tier(df, headset):
    df = df.copy()
    key = list(zip(df["category"], df["cs_gtin"]))
    df["tier"] = ["H" if k in headset else "T" for k in key]
    return df


def fit_grain(fit_df, group_cols):
    """Fit per-group weights (weight_fit) with shrinkage toward the per-category prior."""
    d = fit_df.assign(g=fit_df["s"], l=fit_df["c"])
    d["_grp"] = d[group_cols].astype(str).agg(" | ".join, axis=1)
    kw = dict(grain="gtin", objective="wape_asym", bias_tolerance=0.02, guardrail=True,
              lego_col="rf", gtin_col="cs_gtin", week_col=WEEK)
    grp = fit_per_category_weights(d, cat_col="_grp", **kw)
    cat = fit_per_category_weights(d, cat_col="category", **kw)
    vol = d.groupby("_grp")["actual"].sum()
    g2c = d.drop_duplicates("_grp").set_index("_grp")["category"].to_dict()
    out = {}
    for k, v in grp.items():
        cw = cat.get(g2c.get(k), {"weights": [0.0, 0.0, 1.0], "use": "LEGO"})
        if float(vol.get(k, 0.0)) < MIN_VOL:
            out[k] = {"weights": [0.0, 0.0, 1.0], "use": "LEGO"}; continue
        lam = float(vol.get(k, 0.0)) / (float(vol.get(k, 0.0)) + SHRINK_K)
        w = [lam * v["weights"][i] + (1 - lam) * cw["weights"][i] for i in range(3)]
        use = "STACK" if (v["use"] == "STACK" or cw["use"] == "STACK") else "LEGO"
        out[k] = {"weights": w if use == "STACK" else [0.0, 0.0, 1.0], "use": use}
    return out


def apply_blend(df, fitted, group_cols):
    grp = df[group_cols].astype(str).agg(" | ".join, axis=1).values
    W = np.array([fitted.get(k, {"weights": [0.0, 0.0, 1.0]})["weights"] for k in grp])
    return np.clip(W[:, 0] * df["s"].values + W[:, 1] * df["c"].values + W[:, 2] * df["rf"].values, 0, None)


def score(ev, label):
    def sc(frame, col):
        gg = frame.groupby(["cs_gtin", WEEK]).agg(p=(col, "sum"), a=("actual", "sum")).reset_index()
        return _acc(gg.p, gg.a), _bias(gg.p, gg.a)
    nwin = 0
    cats = sorted(ev["category"].unique())
    deltas = []
    for cat in cats:
        sub = ev[ev["category"] == cat]
        aL, bL = sc(sub, "rf"); aB, bB = sc(sub, "blend")
        win = (aB >= aL - 1e-3) and (min(bB, 0) >= min(bL, 0) - 0.02)
        nwin += int(win); deltas.append(aB - aL)
    aL, bL = sc(ev, "rf"); aB, bB = sc(ev, "blend")
    print(f"  {label:<26} pooled acc {aL:.3f}->{aB:.3f} ({aB-aL:+.3f})  bias {bL:+.3f}->{bB:+.3f}  "
          f"wins {nwin}/{len(cats)}  meanΔ {np.mean(deltas):+.3f}")
    return nwin, aB - aL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="202601,202606,202611,202616")
    ap.add_argument("--eval-snapshot", default="202616")
    ap.add_argument("--nboost", type=int, default=700)
    ap.add_argument("--head-pct", type=float, default=0.2)
    ap.add_argument("--lags", default="4,5")
    a = ap.parse_args()
    snaps = a.snapshots.split(",")
    lags = [int(x) for x in a.lags.split(",")]
    comps = load(snaps, a.nboost, lags)
    fit_snaps = [s for s in snaps if s != a.eval_snapshot]
    fit_all = pd.concat([comps[s] for s in fit_snaps], ignore_index=True)

    # head set from FIT-snapshot volume (leak-free; high-volume gtins are stable)
    vol = fit_all.groupby(["category", "cs_gtin"])["actual"].sum().reset_index()
    headset = set()
    for cat, g in vol.groupby("category"):
        g = g.sort_values("actual", ascending=False)
        n = max(1, int(np.ceil(len(g) * a.head_pct)))
        headset |= set(zip(g.head(n)["category"], g.head(n)["cs_gtin"]))
    fit_t = add_tier(fit_all, headset)
    ev_t = add_tier(comps[a.eval_snapshot], headset)

    print(f"\n===== HEAD-aware blend grains  (fit {fit_snaps} -> eval {a.eval_snapshot}, "
          f"head={a.head_pct:.0%}, lags {lags}, nb{a.nboost}) =====")
    for cols in (["category", "segment"], ["category", "tier"], ["category", "segment", "tier"]):
        fitted = fit_grain(fit_t, cols)
        ev = ev_t.copy()
        ev["blend"] = apply_blend(ev, fitted, cols)
        score(ev, " × ".join(cols))


if __name__ == "__main__":
    main()
