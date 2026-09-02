#!/usr/bin/env python
"""
HEAD lever — pooled LEGO-aware residual model. Trains ONE GBM on the fit snapshots'
forward windows (origin=train_end -> target weeks at lags 4/5) to predict `actual`
from the numeric feature panel PLUS LEGO's own prediction `rf` as a feature (so it can
learn rf + a correction). Applied to the held-out snapshot and scored on the head.

If this LEGO-aware global model can't beat LEGO on the head out-of-sample, no residual
edge exists. Categorical codes differ per snapshot, so we use the consistent NUMERIC
features + rf + raw `category` (fixed-category dtype) only.

    .venv/bin/python scripts/de_residual.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgbm
from de_segment_research import build_panel, KEY, WEEK, CACHE, CONFIG   # noqa: E402
from utils.uk_forecast import add_weeks, load_config, _lgb_params       # noqa: E402

INSCOPE = sorted(["SCRATCH COOKING AIDS", "DEODORANTS & FRAGRANCES", "DRESSINGS", "SKIN CLEANSING",
                  "HEALTHY SNACKING", "HOME & HYGIENE", "FABRIC CLEANING", "ORAL CARE", "SKIN CARE", "HAIR CARE"])
HORIZONS = [4, 5]
FIT_SNAPS = ["202601", "202606", "202611"]
EVAL_SNAP = "202616"


def _acc(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return 1.0 - np.abs(f - a).sum() / max(a.sum(), 1e-9)


def _bias(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return (f - a).sum() / max(a.sum(), 1e-9)


def load_actuals():
    a = pd.read_parquet(CACHE / "actuals" / "actuals_202625.parquet")
    a[KEY] = a[KEY].astype(str)
    a[WEEK] = a[WEEK].astype(str).str.replace("-", "", regex=False)
    return a.rename(columns={"Actuals": "actual"}).groupby([KEY, WEEK], as_index=False)["actual"].sum()


def fwd_rows(snap):
    path, feat, catc, train_end, keymap = build_panel(snap)
    df = pd.read_parquet(path)
    num_feat = [c for c in feat if c not in catc]               # snapshot-consistent numerics
    base = df[df[WEEK] == train_end][[KEY, *num_feat]]
    rows = []
    for h in HORIZONS:
        b = base.copy(); b[WEEK] = add_weeks(train_end, h); b["horizon"] = h
        rows.append(b)
    return pd.concat(rows, ignore_index=True), num_feat


def attach(fr, snap, acts):
    bench = pd.read_parquet(CACHE / "benchmark" / f"bench_{snap}.parquet")
    bench[KEY] = bench[KEY].astype(str)
    bench[WEEK] = bench[WEEK].astype(str).str.replace("-", "", regex=False)
    bench["rf"] = pd.to_numeric(bench["prediction_xgb"], errors="coerce").fillna(0.0)
    b = bench[[KEY, WEEK, "rf", "category", "cs_gtin"]].drop_duplicates([KEY, WEEK])
    return fr.merge(b, on=[KEY, WEEK], how="inner").merge(acts, on=[KEY, WEEK], how="inner")


def main():
    acts = load_actuals()
    fit_parts, num_feat = [], None
    for s in FIT_SNAPS:
        fr, num_feat = fwd_rows(s)
        fit_parts.append(attach(fr, s, acts))
    ev, _ = fwd_rows(EVAL_SNAP)
    ev = attach(ev, EVAL_SNAP, acts)
    FIT = pd.concat(fit_parts, ignore_index=True)
    print(f"train rows {len(FIT):,} (fit {FIT_SNAPS}), eval rows {len(ev):,} ({EVAL_SNAP}); {len(num_feat)} numeric feats + rf + category")

    FEAT = num_feat + ["rf", "category"]
    for d in (FIT, ev):
        d["category"] = pd.Categorical(d["category"], categories=INSCOPE)
    lgb = _lgb_params(load_config(CONFIG)); lgb["num_threads"] = 6
    dtr = lgbm.Dataset(FIT[FEAT], label=np.clip(FIT["actual"].values, 0, None),
                       categorical_feature=["category"], free_raw_data=False)
    model = lgbm.train(lgb, dtr, num_boost_round=700)
    ev["resid_model"] = np.clip(model.predict(ev[FEAT]), 0, None)

    # head set from FIT-snapshot volume
    fitvol = FIT.groupby(["category", "cs_gtin"])["actual"].sum().reset_index()
    headset = set()
    for cat, g in fitvol.groupby("category"):
        g = g.sort_values("actual", ascending=False); n = max(1, int(np.ceil(len(g) * 0.2)))
        headset |= set(zip(g.head(n)["category"], g.head(n)["cs_gtin"]))
    ev["head"] = [(c, gt) in headset for c, gt in zip(ev["category"], ev["cs_gtin"])]
    evh = ev[ev["head"]]

    def sc(frame, col):
        g = frame.groupby(["cs_gtin", WEEK]).agg(p=(col, "sum"), a=("actual", "sum")).reset_index()
        return _acc(g.p, g.a), _bias(g.p, g.a)

    print(f"\n===== HEAD: LEGO vs pooled LEGO-aware residual model (out-of-sample, eval {EVAL_SNAP}) =====")
    print(f"{'category':<26}{'acc_LEGO':>9}{'acc_RESID':>10}{'Δacc':>8}  {'bias_LEGO':>10}{'bias_RESID':>11}  win")
    nwin = 0
    for cat in sorted(evh["category"].dropna().unique()):
        sub = evh[evh["category"] == cat]
        aL, bL = sc(sub, "rf"); aR, bR = sc(sub, "resid_model")
        win = aR > aL + 1e-3; nwin += int(win)
        print(f"{cat:<26}{aL:>9.3f}{aR:>10.3f}{aR-aL:>+8.3f}  {bL:>+10.3f}{bR:>+11.3f}  {'✓' if win else ''}")
    aL, bL = sc(evh, "rf"); aR, bR = sc(evh, "resid_model")
    print(f"{'— HEAD pooled —':<26}{aL:>9.3f}{aR:>10.3f}{aR-aL:>+8.3f}  {bL:>+10.3f}{bR:>+11.3f}")
    print(f"\nResidual model beats LEGO on head: {nwin}/{evh['category'].nunique()} categories")


if __name__ == "__main__":
    main()
