#!/usr/bin/env python
"""
HEAD diagnosis — why does LEGO win on the high-volume head, and is it beatable?
Operates on the CACHED components (instant; no retrain).

For the top-volume GTINs per category it:
  - scores LEGO (rf), category model (c), segment model (s) at GTIN grain (acc + bias),
  - decomposes LEGO's head error into a systematic-BIAS part vs a DISPERSION part,
  - tests a leak-free per-(category) head BIAS CALIBRATION of LEGO (factor learned on the
    fit snapshots, applied to the held-out snapshot).

    .venv/bin/python scripts/de_head_diag.py --nboost 700 --head-pct 0.2
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE = Path("datacache/de/parallel_run")
WEEK = "year_week"


def _acc(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return 1.0 - np.abs(f - a).sum() / max(a.sum(), 1e-9)


def _bias(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return (f - a).sum() / max(a.sum(), 1e-9)


def gtin_agg(frame, col):
    return frame.groupby(["cs_gtin", WEEK]).agg(p=(col, "sum"), a=("actual", "sum")).reset_index()


def load(snaps, nboost, lags):
    acts = pd.read_parquet(CACHE / "actuals" / "actuals_202625.parquet")
    acts["key"] = acts["key"].astype(str)
    acts[WEEK] = acts[WEEK].astype(str).str.replace("-", "", regex=False)
    acts = acts.rename(columns={"Actuals": "actual"}).groupby(["key", WEEK], as_index=False)["actual"].sum()
    out = {}
    for s in snaps:
        c = pd.read_parquet(CACHE / "components" / f"comp_{s}_h4-5_nb{nboost}.parquet")
        c = c[c["horizon"].isin(lags)].merge(acts, on=["key", WEEK], how="inner")
        out[s] = c
    return out


def head_gtins(df, pct):
    """Top-pct GTINs by total actual volume, per category."""
    vol = df.groupby(["category", "cs_gtin"])["actual"].sum().reset_index()
    keep = []
    for cat, g in vol.groupby("category"):
        g = g.sort_values("actual", ascending=False)
        n = max(1, int(np.ceil(len(g) * pct)))
        keep.append(g.head(n)[["category", "cs_gtin"]])
    return pd.concat(keep, ignore_index=True)


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

    # head GTIN set per snapshot
    fit_snaps = [s for s in snaps if s != a.eval_snapshot]
    ev = comps[a.eval_snapshot].copy()
    head = head_gtins(ev, a.head_pct)
    ev_head = ev.merge(head, on=["category", "cs_gtin"], how="inner")

    print(f"\n===== HEAD diagnosis (top {a.head_pct:.0%} GTINs/cat, eval {a.eval_snapshot}, lags {lags}, nb{a.nboost}) =====")
    print(f"{'category':<26}{'%vol':>6}{'acc_LEGO':>9}{'acc_cat':>9}{'acc_seg':>9}  "
          f"{'bias_LEGO':>10}{'|bias|/WAPE':>12}")
    for cat in sorted(ev["category"].unique()):
        full = ev[ev["category"] == cat]
        sub = ev_head[ev_head["category"] == cat]
        if sub.empty:
            continue
        gl = gtin_agg(sub, "rf"); gc = gtin_agg(sub, "c"); gs = gtin_agg(sub, "s")
        aL = _acc(gl.p, gl.a); aC = _acc(gc.p, gc.a); aS = _acc(gs.p, gs.a)
        bL = _bias(gl.p, gl.a)
        wape = np.abs(gl.p - gl.a).sum() / max(gl.a.sum(), 1e-9)
        bias_share = abs(bL) / max(wape, 1e-9)         # how much of LEGO's head error is systematic bias
        head_vol_share = sub["actual"].sum() / max(full["actual"].sum(), 1e-9)
        print(f"{cat:<26}{head_vol_share:>6.0%}{aL:>9.3f}{aC:>9.3f}{aS:>9.3f}  {bL:>+10.3f}{bias_share:>12.0%}")

    # --- leak-free per-category head bias calibration: factor on FIT snaps -> apply to eval head ---
    print(f"\n===== head BIAS-CALIBRATED LEGO  (factor learned on {fit_snaps}, applied to {a.eval_snapshot}) =====")
    fit_all = pd.concat([comps[s] for s in fit_snaps], ignore_index=True)
    fit_head = fit_all.merge(head_gtins(fit_all, a.head_pct), on=["category", "cs_gtin"], how="inner")
    # k_cat = sum(actual)/sum(rf) on the fit head  (so calibrated LEGO is bias-neutral on fit head)
    kf = fit_head.groupby("category").apply(
        lambda g: g["actual"].sum() / max(g["rf"].sum(), 1e-9)).clip(0.5, 1.5)
    print(f"{'category':<26}{'k_cal':>7}{'acc_LEGO':>9}{'acc_calLEGO':>12}{'Δacc':>8}  {'bias_LEGO':>10}{'bias_cal':>10}")
    nwin = 0
    for cat in sorted(ev_head["category"].unique()):
        sub = ev_head[ev_head["category"] == cat]
        k = float(kf.get(cat, 1.0))
        gl = gtin_agg(sub, "rf")
        sub2 = sub.assign(rf_cal=sub["rf"] * k)
        gk = gtin_agg(sub2, "rf_cal")
        aL, aK = _acc(gl.p, gl.a), _acc(gk.p, gk.a)
        bL, bK = _bias(gl.p, gl.a), _bias(gk.p, gk.a)
        nwin += int(aK > aL + 1e-3)
        print(f"{cat:<26}{k:>7.3f}{aL:>9.3f}{aK:>12.3f}{aK-aL:>+8.3f}  {bL:>+10.3f}{bK:>+10.3f}")
    print(f"calibration improves head accuracy on {nwin}/{ev_head['category'].nunique()} categories")


if __name__ == "__main__":
    main()
