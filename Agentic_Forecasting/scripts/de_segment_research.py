#!/usr/bin/env python
"""
Phase 1-3 LOCAL research harness — segment(category × lego_segment) + category + LEGO
stacked ensemble for DE. Standalone (does NOT touch utils/uk_engine.py); reuses the
proven utils.uk_forecast feature pipeline + utils.weight_fit optimiser.

For each training snapshot it builds the DT⋈FS feature panel once, then trains in
parallel:
  - c : per-CATEGORY GBM (today's "local")
  - s : per-(CATEGORY × lego_segment) GBM (the new segment layer; min-rows guard)
captures their forward-week predictions, attaches LEGO rf=prediction_xgb from the
cached benchmark, joins realized actuals (202625), then fits per-(category×segment)
blend weights [w_s, w_c, w_rf] (reusing weight_fit by mapping s->g, c->l and using
category|segment as the group key) and prints a per-category GTIN-grain scoreboard
vs prediction_xgb.

    .venv/bin/python scripts/de_segment_research.py --snapshots 202611,202616 --horizons 4,5
    .venv/bin/python scripts/de_segment_research.py            # all 4 snaps, h=1..5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.uk_forecast import (  # noqa: E402
    load_config, add_weeks, add_backward_features, _encode_categoricals, _future_cols, _lgb_params,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("de_seg")

CACHE = Path("datacache/de/parallel_run")
CONFIG = "config/config_de_base_local.yaml"
TRAIN_SNAPS = ["202601", "202606", "202611", "202616"]
ACTUALS = CACHE / "actuals" / "actuals_202625.parquet"
CAT_COL, SEG_COL, TARGET, KEY, WEEK = "Category", "lego_segment", "Actuals", "key", "year_week"
SEG_MIN_ROWS = 400        # below this a (cat×seg) group is too small to train -> skip (blend falls back to c/rf)
N_BOOST = int(os.environ.get("DE_SEG_NBOOST", "300"))   # research speed (prod uses 700); DE_SEG_NBOOST=700
RECENCY_HL = 39.0


# --------------------------------------------------------------------------- #
# Feature panel (reuses the engine's helpers; mirrors uk_engine._prepare_state)
# --------------------------------------------------------------------------- #
def build_panel(snap: str):
    dt = pd.read_parquet(CACHE / "dt" / f"dt_{snap}.parquet")
    fs = pd.read_parquet(CACHE / "fs" / f"fs_{snap}.parquet")
    add = [c for c in fs.columns if c == KEY or c == WEEK or c not in dt.columns]
    df = dt.merge(fs[add], on=[KEY, WEEK], how="inner")          # inner => exclude_forecast==0 universe
    df[WEEK] = df[WEEK].astype(str).str.replace("-", "", regex=False)
    df[KEY] = df[KEY].astype(str)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0)
    df = df.sort_values([KEY, WEEK], kind="mergesort").reset_index(drop=True)
    df["_cat_raw"] = df[CAT_COL].astype(str)
    df["_seg_raw"] = df[SEG_COL].astype(str)

    cfg = load_config(CONFIG, overrides={"input_source": "dbx", "collapse_promo": False,
                                         "holiday_distance_features": False, "rich_features": False,
                                         "per_category": False, "bias_calibration": False})
    cfg = {**cfg, "cat_col": CAT_COL}
    back = add_backward_features(df, cfg)
    catc = _encode_categoricals(df, cfg)        # encodes Category/Brand/... to int codes IN PLACE
    futc = _future_cols(df, cfg)
    feat = list(dict.fromkeys(back + catc + futc))
    weeks = sorted(df[WEEK].unique())
    df["_wi"] = df[WEEK].map({w: i for i, w in enumerate(weeks)}).astype("int32")

    path = str(Path(tempfile.gettempdir()) / f"de_seg_panel_{snap}.parquet")
    keep = list(dict.fromkeys([KEY, WEEK, "_wi", TARGET, "_cat_raw", "_seg_raw", *feat]))
    df[keep].to_parquet(path, index=False)
    train_end = add_weeks(snap, -1)
    # key -> (cat, seg) map for this snapshot
    keymap = df[[KEY, "_cat_raw", "_seg_raw"]].drop_duplicates(KEY)
    logger.info("[panel] %s: %d rows, %d feats, train_end=%s, %d cats, %d cat×seg groups",
                snap, len(df), len(feat), train_end,
                df["_cat_raw"].nunique(), df.groupby("_cat_raw")["_seg_raw"].nunique().sum())
    return path, feat, catc, train_end, keymap


# --------------------------------------------------------------------------- #
# Worker: train one group (category OR category×segment) for one horizon
# --------------------------------------------------------------------------- #
def _train_one(args):
    (kind, h, cat, seg, panel_path, train_end, feat, catc, lgb) = args
    import lightgbm as lgbm
    cols = list({KEY, WEEK, TARGET, "_cat_raw", "_seg_raw", "_wi", *feat})
    df = pd.read_parquet(panel_path, columns=cols)
    if kind == "category":
        df = df[df["_cat_raw"] == cat]
    else:
        df = df[(df["_cat_raw"] == cat) & (df["_seg_raw"] == seg)]
    empty = {"kind": kind, "cat": cat, "seg": seg, "h": h, "pred": None}
    if len(df) < (SEG_MIN_ROWS if kind == "segment" else 50):
        return empty
    tw = {w: add_weeks(w, h) for w in df[WEEK].unique()}
    left = df[[KEY, WEEK, "_wi", *feat]].copy()
    left["_tw"] = left[WEEK].map(tw)
    right = df[[KEY, WEEK, TARGET]].rename(columns={WEEK: "_tw", TARGET: "_y"})
    pairs = left.merge(right, on=[KEY, "_tw"], how="inner")
    tr = pairs[pairs["_tw"] <= train_end]
    pr = pairs[pairs[WEEK] == train_end]
    if tr.empty or pr.empty:
        return empty
    te = tr["_wi"].max()
    w = np.power(0.5, (te - tr["_wi"]).clip(lower=0) / RECENCY_HL)
    dtrain = lgbm.Dataset(tr[feat], label=np.clip(tr["_y"].values, 0, None), weight=w.values,
                          categorical_feature=catc, free_raw_data=False)
    model = lgbm.train(lgb, dtrain, num_boost_round=N_BOOST)
    out = pr[[KEY, "_tw"]].rename(columns={"_tw": WEEK}).copy()
    out["value"] = np.clip(model.predict(pr[feat]), 0, None)
    out["horizon"] = h
    return {"kind": kind, "cat": cat, "seg": seg, "h": h, "pred": out}


# --------------------------------------------------------------------------- #
# Components for one snapshot: c, s, rf per (key, year_week, horizon)
# --------------------------------------------------------------------------- #
RECOMPUTE = False   # set by --recompute; otherwise reuse cached components (deterministic)


def components(snap: str, horizons, workers: int) -> pd.DataFrame:
    hsig = "-".join(str(h) for h in sorted(horizons))
    cpath = CACHE / "components" / f"comp_{snap}_h{hsig}_nb{N_BOOST}.parquet"
    if cpath.exists() and not RECOMPUTE:
        logger.info("[%s] reuse cached components -> %s", snap, cpath.name)
        return pd.read_parquet(cpath)
    path, feat, catc, train_end, keymap = build_panel(snap)
    g0 = pd.read_parquet(path, columns=[KEY, "_cat_raw", "_seg_raw"])
    cats = sorted(g0["_cat_raw"].unique())
    seg_by_cat = {c: sorted(g0[g0["_cat_raw"] == c]["_seg_raw"].unique()) for c in cats}
    cfg = load_config(CONFIG, overrides={"per_category": False})
    lgb = _lgb_params({**cfg, "cat_col": CAT_COL})
    lgb["num_threads"] = int(os.environ.get("DE_SEG_THREADS", "4"))

    tasks = []
    for h in horizons:
        for c in cats:
            tasks.append(("category", h, c, None, path, train_end, feat, catc, dict(lgb)))
            for s in seg_by_cat[c]:
                tasks.append(("segment", h, c, s, path, train_end, feat, catc, dict(lgb)))
    logger.info("[%s] %d train tasks across %d workers", snap, len(tasks), workers)

    cat_pred, seg_pred = {}, {}
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = [pool.submit(_train_one, t) for t in tasks]
        done = 0
        for f in as_completed(futs):
            r = f.result(); done += 1
            if r["pred"] is None:
                continue
            if r["kind"] == "category":
                cat_pred[(r["cat"], r["h"])] = r["pred"]
            else:
                seg_pred[(r["cat"], r["seg"], r["h"])] = r["pred"]
            if done % 50 == 0:
                logger.info("  [%s] %d/%d tasks", snap, done, len(tasks))

    # assemble: c (category) keyed by key+week+horizon
    c_df = pd.concat([d.assign(category=c) for (c, h), d in cat_pred.items()], ignore_index=True) \
        .rename(columns={"value": "c"}) if cat_pred else pd.DataFrame()
    s_df = pd.concat([d for d in seg_pred.values()], ignore_index=True) \
        .rename(columns={"value": "s"}) if seg_pred else pd.DataFrame(columns=[KEY, WEEK, "horizon", "s"])

    comp = c_df[[KEY, WEEK, "horizon", "c"]].merge(
        s_df[[KEY, WEEK, "horizon", "s"]], on=[KEY, WEEK, "horizon"], how="left")
    # LEGO rf from benchmark
    bench = pd.read_parquet(CACHE / "benchmark" / f"bench_{snap}.parquet")
    bench[KEY] = bench[KEY].astype(str)
    bench[WEEK] = bench[WEEK].astype(str).str.replace("-", "", regex=False)
    bench["rf"] = pd.to_numeric(bench["prediction_xgb"], errors="coerce").fillna(0.0)
    comp = comp.merge(bench[[KEY, WEEK, "rf"]].drop_duplicates([KEY, WEEK]), on=[KEY, WEEK], how="left")
    comp = comp.merge(keymap, on=KEY, how="left")
    comp["category"] = comp["_cat_raw"]; comp["segment"] = comp["_seg_raw"]
    comp["cs_gtin"] = comp[KEY].str.rsplit("_", n=1).str[0]
    comp["snapshot"] = snap
    comp["s"] = comp["s"].fillna(comp["c"])      # tiny/absent segment -> fall back to category model
    comp["rf"] = comp["rf"].fillna(0.0)
    out = comp[[KEY, WEEK, "horizon", "s", "c", "rf", "category", "segment", "cs_gtin", "snapshot"]]
    cpath.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cpath, index=False)
    return out


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
def _acc(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return 1.0 - np.abs(f - a).sum() / max(a.sum(), 1e-9)


def _bias(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return (f - a).sum() / max(a.sum(), 1e-9)


MIN_SEG_VOLUME = 1000.0   # a (cat×seg) group must hold this much realized volume to deploy a stack
SHRINK_K = 5000.0         # volume at which a segment is shrunk 50% toward its category's blend


def fit_segments(fit_df: pd.DataFrame, shrink: bool = True) -> dict:
    """Fit per-(category × lego_segment) weights [w_s, w_c, w_rf] (reuse weight_fit by
    mapping s->g, c->l). HIERARCHICAL SHRINKAGE: each segment's weights are pulled toward
    its CATEGORY-level blend by volume-reliability λ = vol/(vol+K) — sparse segments borrow
    the category blend (which is itself LEGO-guardrailed), so noisy segments still deploy a
    sound blend instead of falling all the way back to LEGO. Tiny-volume groups -> LEGO."""
    from utils.weight_fit import fit_per_category_weights
    d = fit_df.assign(g=fit_df["s"], l=fit_df["c"],
                      catseg=fit_df["category"] + " | " + fit_df["segment"], cat=fit_df["category"])
    kw = dict(grain="gtin", objective="wape_asym", bias_tolerance=0.02, guardrail=True,
              lego_col="rf", gtin_col="cs_gtin", week_col=WEEK)
    seg = fit_per_category_weights(d, cat_col="catseg", **kw)
    cat = fit_per_category_weights(d, cat_col="cat", **kw)          # category-level prior
    vol = d.groupby("catseg")["actual"].sum()

    out = {}
    for k, v in seg.items():
        c = k.split(" | ")[0]
        cw = cat.get(c, {"weights": [0.0, 0.0, 1.0], "use": "LEGO"})
        if float(vol.get(k, 0.0)) < MIN_SEG_VOLUME:
            out[k] = {**v, "weights": [0.0, 0.0, 1.0], "use": "LEGO(lowvol)"}
            continue
        if not shrink:
            out[k] = v
            continue
        lam = float(vol.get(k, 0.0)) / (float(vol.get(k, 0.0)) + SHRINK_K)
        w = [lam * v["weights"][i] + (1.0 - lam) * cw["weights"][i] for i in range(3)]
        # deploy the shrunk blend whenever the segment OR its category beats LEGO; else LEGO
        use = "STACK" if (v["use"] == "STACK" or cw["use"] == "STACK") else "LEGO"
        out[k] = {**v, "weights": (w if use == "STACK" else [0.0, 0.0, 1.0]), "use": use}
    return out


def apply_blend(df: pd.DataFrame, fitted: dict) -> np.ndarray:
    wmap = {k: v["weights"] for k, v in fitted.items()}
    cs = (df["category"] + " | " + df["segment"]).values
    W = np.array([wmap.get(k, [0.0, 0.0, 1.0]) for k in cs])
    return np.clip(W[:, 0] * df["s"].values + W[:, 1] * df["c"].values + W[:, 2] * df["rf"].values, 0, None)


def _score(frame, col):
    g = frame.groupby(["cs_gtin", WEEK]).agg(p=(col, "sum"), a=("actual", "sum")).reset_index()
    return _acc(g.p, g.a), _bias(g.p, g.a)


def scoreboard(df: pd.DataFrame, label: str) -> int:
    print(f"\n================  {label}  ================")
    print(f"{'category':<26}{'acc_LEGO':>9}{'acc_BLEND':>10}{'Δacc':>8}   {'bias_LEGO':>10}{'bias_BLEND':>11}  win")
    nwin = 0
    cats = sorted(df["category"].unique())
    for cat in cats:
        sub = df[df["category"] == cat]
        aL, bL = _score(sub, "rf"); aB, bB = _score(sub, "blend")
        win = (aB >= aL - 1e-3) and (min(bB, 0) >= min(bL, 0) - 0.02)
        nwin += int(win)
        print(f"{cat:<26}{aL:>9.3f}{aB:>10.3f}{aB-aL:>+8.3f}   {bL:>+10.3f}{bB:>+11.3f}  {'✓' if win else ''}")
    # pooled all-category line
    aL, bL = _score(df, "rf"); aB, bB = _score(df, "blend")
    print(f"{'— ALL (pooled) —':<26}{aL:>9.3f}{aB:>10.3f}{aB-aL:>+8.3f}   {bL:>+10.3f}{bB:>+11.3f}")
    print(f"BLEND beats LEGO (acc + asym-bias): {nwin}/{len(cats)} categories")
    return nwin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default=",".join(TRAIN_SNAPS))
    ap.add_argument("--horizons", default="1,2,3,4,5")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) // 2))
    ap.add_argument("--eval-lags", default="4,5", help="horizons to score in the scoreboard")
    ap.add_argument("--eval-snapshot", default="", help="held-out snapshot for out-of-sample eval (default: last)")
    ap.add_argument("--recompute", action="store_true", help="retrain models (ignore cached components)")
    a = ap.parse_args()
    global RECOMPUTE
    RECOMPUTE = a.recompute
    snaps = [s.strip().replace("-", "") for s in a.snapshots.split(",") if s.strip()]
    horizons = [int(h) for h in a.horizons.split(",")]
    eval_lags = [int(x) for x in a.eval_lags.split(",")]

    acts = pd.read_parquet(ACTUALS)
    acts[KEY] = acts[KEY].astype(str)
    acts[WEEK] = acts[WEEK].astype(str).str.replace("-", "", regex=False)
    acts = acts.rename(columns={"Actuals": "actual"})[[KEY, WEEK, "actual"]] \
        .groupby([KEY, WEEK], as_index=False)["actual"].sum()

    comps = {}
    for s in snaps:
        c = components(s, horizons, a.workers)
        c = c[c["horizon"].isin(eval_lags)].merge(acts, on=[KEY, WEEK], how="inner")
        comps[s] = c
        logger.info("[%s] %d component rows joined to actuals", s, len(c))

    # --- OUT-OF-SAMPLE: fit weights on the other snapshots, score the held-out one ---
    eval_snap = (a.eval_snapshot.replace("-", "") or snaps[-1])
    fit_snaps = [s for s in snaps if s != eval_snap]
    if fit_snaps:
        fitted = fit_segments(pd.concat([comps[s] for s in fit_snaps], ignore_index=True))
        ev = comps[eval_snap].copy()
        ev["blend"] = apply_blend(ev, fitted)
        scoreboard(ev, f"OUT-OF-SAMPLE  fit{fit_snaps} -> eval {eval_snap}  (GTIN grain, lags {eval_lags})")
        nstk = sum(1 for v in fitted.values() if v["use"] == "STACK")
        print(f"STACK deployed for {nstk}/{len(fitted)} segments (fit on {fit_snaps}).")

    # --- IN-SAMPLE reference (all snapshots pooled) ---
    alldf = pd.concat(comps.values(), ignore_index=True)
    fitted_all = fit_segments(alldf)
    alldf["blend"] = apply_blend(alldf, fitted_all)
    scoreboard(alldf, f"IN-SAMPLE  all{snaps} pooled  (GTIN grain, lags {eval_lags})")
    top = sorted(((k, v) for k, v in fitted_all.items() if v["use"] == "STACK"),
                 key=lambda kv: kv[1]["acc_stack"] - kv[1]["acc_lego"], reverse=True)[:12]
    print("\nTop segment wins (acc_lego -> acc_stack, weights [s,c,rf]):")
    for k, v in top:
        print(f"  {k:<48} {v['acc_lego']:.3f}->{v['acc_stack']:.3f}  w={v['weights']}")


if __name__ == "__main__":
    main()
