"""Option 1: seasonal-level per-key models for high-volume keys, to close the bias gap LEGO wins.
Uses hf.perkey_dmh_config (SPLY-blend 0.4 + recency 0 = full history) so each high-vol key's
SEASONAL level (same-period-last-year) is captured, not just the recent level. Overlays onto the
accuracy-first base (final_forecast_best), evals per category base-vs-overlay at t45 & 13wk, and
assembles a SELECTIVE overlay keeping the seasonal per-key only where it improves the category vs
LEGO. No torch (LightGBM only) -> safe in one process. Saves final_forecast_seasonal_overlay.parquet."""
import os, sys, glob, time
for k, v in {"MLFLOW_AUTOLOGGING_DISABLE": "1", "OTEL_SDK_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.ERROR)
import numpy as np, pandas as pd
import utils.hybrid_forecast as hf
from utils import lego_eval

CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]
ID = ["key", "year_week", "Actuals", "segment_id"]
PAT = ["promo", "discount", "holiday", "songkran", "bucha", "festive", "season", "summer", "winter",
       "monsoon", "week_of_year", "is_pre", "is_post", "weeks_to", "stringency", "off_invoice", "baseline", "sp_week", "qtr_cycle"]
ORIGIN = 202602
VOL_Q = float(os.environ.get("VOL_Q", "0.70"))


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
    cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
    # high-volume keys per category from precomputed features (no full-panel load)
    F = pd.read_parquet("artifacts_th_local/_key_features.parquet"); F["key"] = F["key"].astype(str)
    F["cat"] = F["key"].str[:-6].map(cs2cat); F["zf"] = 1.0 - F["nzfrac"]; F["nz"] = F["nzfrac"] * F["n"]
    hv_by_cat = {}
    for c in sorted(set(cs2cat.values)):
        if c == "ICE CREAM":
            continue
        sub = F[F.cat == c]
        if sub.empty:
            continue
        thr = sub["vol"].quantile(VOL_Q)
        hv_by_cat[c] = sub[(sub.vol >= thr) & (sub.zf <= 0.35) & (sub.nz >= 52)]["key"].tolist()
    print(f"[seasonal] high-vol keys: {sum(len(v) for v in hv_by_cat.values())} across {len(hv_by_cat)} cats (VOL_Q={VOL_Q})", flush=True)

    cfg = hf.perkey_dmh_config()  # SPLY-blend 0.4, recency 0 -> seasonal level
    CATDIR = {c.replace(" & ", "_AND_").replace(" ", "_"): c for c in hv_by_cat}  # eval-name -> not needed; load by dir
    out = []
    for cdir in CATS:
        cname = {"DEODORANTS_AND_FRAGRANCES": "DEODORANTS & FRAGRANCES", "FABRIC_CLEANING": "FABRIC CLEANING",
                 "FABRIC_ENHANCERS": "FABRIC ENHANCERS", "FOODS": "FOODS", "HAIR_CARE": "HAIR CARE",
                 "HOME_AND_HYGIENE": "HOME & HYGIENE", "SKIN_CLEANSING": "SKIN CLEANSING", "BODY": "BODY", "FACE": "FACE"}.get(cdir, cdir)
        hv = set(hv_by_cat.get(cname, []))
        fd = f"artifacts_th_local/final/{cdir}/feature_output"
        if not hv or not glob.glob(f"{fd}/train_features*"):
            continue
        train, numc = _load(fd, "train_features", keys=hv)
        cc = sorted(set(train["key"].astype(str)) & hv)
        if not cc:
            continue
        val, _ = _load(fd, "val_features", keys=hv, keep=numc); test, _ = _load(fd, "test_features", keys=hv, keep=numc)
        fwd = [x for x in train.columns if x not in ("key", "year_week", "Actuals") and any(p in x.lower() for p in PAT)]
        train, val, test = hf.augment_panels(train, val, test, "Actuals", fwd)
        print(f"[seasonal] {cdir}: {len(cc)} hv keys ({time.time()-t0:.0f}s)", flush=True)
        fc = hf.train_perkey_forecasts(train, val, test, cc, cfg, int(ORIGIN))
        if not fc.empty:
            out.append(fc[["key", "year_week", "predicted"]])
    seasonal = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["key", "year_week", "predicted"])
    seasonal["key"] = seasonal["key"].astype(str); seasonal["year_week"] = seasonal["year_week"].astype(int)
    seasonal.to_parquet("artifacts_th_local/_seasonal_perkey_fc.parquet", index=False)
    print(f"[seasonal] trained {seasonal['key'].nunique()} keys ({time.time()-t0:.0f}s)", flush=True)

    # ---- per-category eval: base vs full overlay (all hv keys -> seasonal per-key) ----
    bestfc = pd.read_parquet("artifacts_th_local/final_forecast_best.parquet"); bestfc["key"] = bestfc["key"].astype(str); bestfc["year_week"] = bestfc["year_week"].astype(int)
    got = set(seasonal["key"])
    overlay = pd.concat([seasonal, bestfc[~bestfc["key"].isin(got)]], ignore_index=True).groupby(["key", "year_week"], as_index=False)["predicted"].first()
    pb = lego_eval.attach_forecast(base, bestfc, pred_col="predicted"); pb = pb[pb.Category != "ICE CREAM"]
    po = lego_eval.attach_forecast(base, overlay, pred_col="predicted"); po = po[po.Category != "ICE CREAM"]
    CATSN = sorted(pb.Category.unique())
    print("=== per-category: BASE vs +SEASONAL-OVERLAY (t45 acc/bias, LEGO) ===")
    keep_cats = []
    for c in CATSN:
        b45 = pb[(pb.Category == c) & pb.t.isin([4, 5])]; o45 = po[(po.Category == c) & po.t.isin([4, 5])]
        ba, bb = ab(b45); oa, ob = ab(o45); la, lb = ab(b45.assign(ours=b45.lego))
        b_both = (ba > la) and (abs(bb) < abs(lb)); o_both = (oa > la) and (abs(ob) < abs(lb))
        # keep overlay if it improves the LEGO standing: gains a both-win, or improves bias without losing the acc-win
        keep = o_both and not b_both
        keep = keep or (oa > la and abs(ob) < abs(bb) and (b_both == o_both))
        if keep:
            keep_cats.append(c)
        mark = "KEEP" if keep else "skip"
        print(f"  {c:22s} base a{ba:+.3f} b{bb:+.2f}{'*' if b_both else ' '}  -> overlay a{oa:+.3f} b{ob:+.2f}{'*' if o_both else ' '}  (LEGO a{la:+.3f} b{lb:+.2f})  {mark}", flush=True)

    # ---- selective overlay: seasonal per-key only in KEEP categories ----
    keep_keys = set(seasonal.merge(base[["cs", "Category"]].drop_duplicates().assign(key=lambda d: d["cs"] + "_E1016"),
                                   on="key", how="left").query("Category in @keep_cats")["key"]) if keep_cats else set()
    sel = pd.concat([seasonal[seasonal.key.isin(keep_keys)], bestfc[~bestfc.key.isin(keep_keys)]], ignore_index=True)
    sel = sel.groupby(["key", "year_week"], as_index=False)["predicted"].first()
    sel.to_parquet("artifacts_th_local/final_forecast_seasonal_overlay.parquet", index=False)
    print(f"[seasonal] KEEP categories: {keep_cats}", flush=True)

    def board(fc, name):
        p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
        for tag, ts in [("t45", [4, 5]), ("13wk", list(range(1, 14)))]:
            q = p[p.t.isin(ts)]; rows = []
            for c in CATSN:
                pc = q[q.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego)); rows.append((pc.actual.sum(), oa, la, ob, lb))
            d = pd.DataFrame(rows, columns=["v", "oa", "la", "ob", "lb"]); w = d.v / d.v.sum()
            print("  %s %-4s acc %.3f (LEGO %.3f) bias %+.3f (LEGO %+.3f) | acc-win %d/9 bias-win %d/9 BOTH %d/9" % (
                name, tag, (d.oa * w).sum(), (d.la * w).sum(), (d.ob * w).sum(), (d.lb * w).sum(),
                int((d.oa > d.la).sum()), int((d.ob.abs() < d.lb.abs()).sum()), int(((d.oa > d.la) & (d.ob.abs() < d.lb.abs())).sum())))
    print("=== SCORECARDS ===")
    board(bestfc, "BASE    ")
    board(sel, "SEASONAL")


if __name__ == "__main__":
    main()
