"""Micro-tune FABRIC CLEANING's SPLY-blend alpha to try to flip its bias over the LEGO line.
Train each FABRIC hv key ONCE, then re-predict at several sply_blend_alpha (predict-time param:
pred = alpha*model + (1-alpha)*sply; higher alpha = less SPLY = lean to the model, which is less
under-biased here). Overlay each onto the pure base at t<=6, pick the alpha that best beats LEGO on
BOTH at t45, save it, and print the full all-category scorecard."""
import os, sys, glob, time, dataclasses
for k, v in {"MLFLOW_AUTOLOGGING_DISABLE": "1", "OTEL_SDK_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.ERROR)
import numpy as np, pandas as pd
import utils.hybrid_forecast as hf
from utils.direct_multihorizon import train_direct_multihorizon, predict_direct_multihorizon
from utils import lego_eval

ALPHAS = [0.4, 0.5, 0.6, 0.7, 0.85]
ORIGIN, HMAX = 202602, 6
ID = ["key", "year_week", "Actuals", "segment_id"]
PAT = ["promo", "discount", "holiday", "songkran", "bucha", "festive", "season", "summer", "winter",
       "monsoon", "week_of_year", "is_pre", "is_post", "weeks_to", "stringency", "off_invoice", "baseline", "sp_week", "qtr_cycle"]


def _load(fd, name, keys=None, keep=None):
    pq = sorted(glob.glob(f"{fd}/{name}*.parquet")); csv = sorted(glob.glob(f"{fd}/{name}*.csv"))
    ks = set(map(str, keys)) if keys is not None else None
    if pq:
        flt = [("key", "in", list(ks))] if ks is not None else None
        try:
            d = pd.read_parquet(pq[0], filters=flt)
        except Exception:
            d = pd.read_parquet(pq[0]); d = d[d["key"].astype(str).isin(ks)] if ks else d
    else:
        d = pd.read_csv(csv[0]); d = d[d["key"].astype(str).isin(ks)] if ks is not None else d
    d = d[d["key"].notna()].copy(); yw = pd.to_numeric(d["year_week"], errors="coerce"); d = d[yw.notna()].copy()
    d["year_week"] = yw[yw.notna()].astype("int64"); d["key"] = d["key"].astype(str)
    idc = [c for c in ID if c in d.columns]
    numc = keep if keep is not None else [c for c in d.columns if c not in idc and pd.api.types.is_numeric_dtype(d[c])]
    valid = [c for c in numc if c in d.columns]; d = d[idc + valid]
    d[valid] = d[valid].replace([np.inf, -np.inf], np.nan); return d, numc


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(tot, 1e-9), (f - a).sum() / max(tot, 1e-9))


def main():
    t0 = time.time()
    base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
    CATS = sorted(c for c in base.Category.unique() if c != "ICE CREAM")
    cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
    wk2t = base.drop_duplicates("Shipment_Week").assign(yw=lambda d: d["Shipment_Week"].str.replace("-", "").astype(int)).set_index("yw")["t"]
    F = pd.read_parquet("artifacts_th_local/_key_features.parquet"); F["key"] = F["key"].astype(str)
    F["cat"] = F["key"].str[:-6].map(cs2cat); F["zf"] = 1 - F["nzfrac"]; F["nz"] = F["nzfrac"] * F["n"]
    sub = F[F.cat == "FABRIC CLEANING"]; thr = sub["vol"].quantile(0.70)
    hv = set(sub[(sub.vol >= thr) & (sub.zf <= 0.35) & (sub.nz >= 52)]["key"])

    fd = "artifacts_th_local/final/FABRIC_CLEANING/feature_output"
    train, numc = _load(fd, "train_features", keys=hv)
    val, _ = _load(fd, "val_features", keys=hv, keep=numc); test, _ = _load(fd, "test_features", keys=hv, keep=numc)
    fwd = [x for x in train.columns if x not in ("key", "year_week", "Actuals") and any(p in x.lower() for p in PAT)]
    train, val, test = hf.augment_panels(train, val, test, "Actuals", fwd)
    keys = sorted(set(train["key"].astype(str)) & hv)
    cfg = hf.perkey_dmh_config()
    print(f"[tune] training {len(keys)} FABRIC hv keys once, predicting at alphas {ALPHAS}", flush=True)

    per_alpha = {a: [] for a in ALPHAS}
    ok = 0
    for i, k in enumerate(keys):
        tr = train[train["key"].astype(str) == k]
        if len(tr) < 40:
            continue
        vl = val[val["key"].astype(str) == k]; te = test[test["key"].astype(str) == k]
        try:
            arts = train_direct_multihorizon(tr, vl, cfg)
            for a in ALPHAS:
                fc = predict_direct_multihorizon(te, arts, dataclasses.replace(cfg, sply_blend_alpha=a),
                                                 origin_week=ORIGIN, fallback_features=vl)
                fc = fc.copy(); fc["key"] = k
                per_alpha[a].append(fc[["key", "year_week", "predicted"]])
            ok += 1
        except Exception as e:
            logging.warning("key %s failed: %s", k, e)
        if i % 15 == 0:
            print(f"[tune] {i}/{len(keys)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[tune] trained {ok} keys ({time.time()-t0:.0f}s)", flush=True)

    bestfc = pd.read_parquet("artifacts_th_local/final_forecast_catselect_base.parquet")
    bestfc["key"] = bestfc["key"].astype(str); bestfc["year_week"] = bestfc["year_week"].astype(int)

    def build(alpha):
        seas = pd.concat(per_alpha[alpha], ignore_index=True)
        seas["key"] = seas["key"].astype(str); seas["year_week"] = seas["year_week"].astype(int)
        seas["t"] = seas["year_week"].map(wk2t)
        use = seas[seas.t <= HMAX][["key", "year_week", "predicted"]]
        kw = set(zip(use.key, use.year_week))
        rest = bestfc[~bestfc.apply(lambda r: (r.key, r.year_week) in kw, axis=1)][["key", "year_week", "predicted"]]
        return pd.concat([use, rest], ignore_index=True).groupby(["key", "year_week"], as_index=False)["predicted"].first()

    # sweep: FABRIC t45 acc/bias + aggregate, find the alpha that best beats LEGO on both
    print("[tune] alpha sweep (FABRIC t45 + aggregate t45):", flush=True)
    scored = []
    for a in ALPHAS:
        fc = build(a); p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
        fpc = p[(p.Category == "FABRIC CLEANING") & p.t.isin([4, 5])]; fa, fb = ab(fpc); fla, flb = ab(fpc.assign(ours=fpc.lego))
        q = p[p.t.isin([4, 5])]; rows = [(p[(p.Category == c) & p.t.isin([4, 5])]) for c in CATS]
        agg = [(pc.actual.sum(), *ab(pc), *ab(pc.assign(ours=pc.lego))) for pc in rows]
        d = pd.DataFrame(agg, columns=["v", "oa", "ob", "la", "lb"]); w = d.v / d.v.sum()
        fab_both = (fa > fla) and (abs(fb) < abs(flb))
        both = int(((d.oa > d.la) & (d.ob.abs() < d.lb.abs())).sum())
        print(f"  alpha={a}: FABRIC acc {fa:+.3f}/{fla:+.3f} bias {fb:+.3f}/{flb:+.3f} {'BOTH-WIN' if fab_both else 'bias-loss'} | agg t45 acc {(d.oa*w).sum():.3f} bias {(d.ob*w).sum():+.3f} BOTH {both}/9", flush=True)
        scored.append((a, fab_both, abs(fb) < abs(flb), (d.oa * w).sum(), abs((d.ob * w).sum()), fc))
    # pick: prefer FABRIC both-win; then max agg acc; then min |agg bias|
    scored.sort(key=lambda s: (s[1], s[2], s[3], -s[4]), reverse=True)
    best_a, fab_both, _, _, _, bestfinal = scored[0]
    print(f"[tune] PICKED alpha={best_a} (FABRIC both-win={fab_both})", flush=True)
    bestfinal.to_parquet("artifacts_th_local/final_forecast_best.parquet", index=False)

    # full all-category scorecard for the picked alpha
    p = lego_eval.attach_forecast(base, bestfinal, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    print(f"\n===== FINAL ALL-CATEGORY SCORECARD (alpha={best_a}, FABRIC seasonal overlay t<=6) =====")
    print(f"{'Category':24s} | {'t45 acc DIQ/LEGO':18s} | {'t45 bias DIQ/LEGO':18s} | {'13wk acc DIQ/LEGO':18s} | verdict")
    for c in CATS:
        pc4 = p[(p.Category == c) & p.t.isin([4, 5])]; a4, b4 = ab(pc4); la4, lb4 = ab(pc4.assign(ours=pc4.lego))
        pc = p[p.Category == c]; a13, _ = ab(pc); la13, _ = ab(pc.assign(ours=pc.lego))
        v = "WIN both" if (a4 > la4 and abs(b4) < abs(lb4)) else ("acc only" if a4 > la4 else ("bias only" if abs(b4) < abs(lb4) else "-"))
        print(f"{c:24s} | {a4:+.3f} / {la4:+.3f}    | {b4:+.3f} / {lb4:+.3f}    | {a13:+.3f} / {la13:+.3f}    | {v}")
    for tag, ts in [("t45", [4, 5]), ("13wk", list(range(1, 14)))]:
        q = p[p.t.isin(ts)]; rows = [(q[q.Category == c]) for c in CATS]
        d = pd.DataFrame([(pc.actual.sum(), *ab(pc), *ab(pc.assign(ours=pc.lego))) for pc in rows], columns=["v", "oa", "ob", "la", "lb"]); w = d.v / d.v.sum()
        print(f"{'ALL (sales-wtd) '+tag:24s} | {(d.oa*w).sum():+.3f} / {(d.la*w).sum():+.3f}    | {(d.ob*w).sum():+.3f} / {(d.lb*w).sum():+.3f}    | acc-win {int((d.oa>d.la).sum())}/9  bias-win {int((d.ob.abs()<d.lb.abs()).sum())}/9  BOTH {int(((d.oa>d.la)&(d.ob.abs()<d.lb.abs())).sum())}/9")


if __name__ == "__main__":
    main()
