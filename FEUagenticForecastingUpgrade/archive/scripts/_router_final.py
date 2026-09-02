"""Final forecast from the router assignment: OWN per-key model for the assigned keys
(captures spiky high-volume keys at their true level), bias-fixed POOLED deep model for the
rest. No torch in this process (loads the saved deep test forecasts). Builds
final_forecast_router.parquet, evals vs LEGO, and reports per category (acc + bias + win)."""
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


def main():
    t0 = time.time()
    asg = pd.read_parquet("artifacts_th_local/_router_assignment.parquet")
    own = set(asg.loc[asg.own == 1, "key"].astype(str))
    print(f"[final] {len(own)} OWN per-key keys, {len(asg)-len(own)} pooled", flush=True)

    # 1. POOLED deep (bias-fixed = Huber8 shape x RMSE cat-level), for ALL keys (override own below)
    base = lego_eval.load_eval_base(); cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
    H = pd.read_parquet("artifacts_th_local/deep_TFT_Huber8_fc.parquet").rename(columns={"predicted": "h"})
    R = pd.read_parquet("artifacts_th_local/deep_TFT_RMSE_fc.parquet").rename(columns={"predicted": "r"})
    m = H.merge(R, on=["key", "year_week"]); m["Category"] = m["key"].astype(str).str[:-6].map(cs2cat)
    catf = m.groupby("Category").apply(lambda d: d["r"].sum() / max(d["h"].sum(), 1e-9))
    m["predicted"] = (m["h"] * m["Category"].map(catf).fillna(1.0)).clip(lower=0)
    pooled = m[["key", "year_week", "predicted"]].copy()

    # 2. OWN per-key forecast (test origin 202602) for assigned keys
    perkey = []
    for c in CATS:
        fd = f"artifacts_th_local/final/{c}/feature_output"
        if not glob.glob(f"{fd}/train_features*"):
            continue
        train, numc = _load(fd, "train_features", keys=own)
        cc = sorted(set(train["key"].astype(str)) & own)
        if not cc:
            continue
        val, _ = _load(fd, "val_features", keys=own, keep=numc); test, _ = _load(fd, "test_features", keys=own, keep=numc)
        fwd = [x for x in train.columns if x not in ("key", "year_week", "Actuals") and any(p in x.lower() for p in PAT)]
        train, val, test = hf.augment_panels(train, val, test, "Actuals", fwd)
        print(f"[final] {c}: per-key forecasting {len(cc)} own keys ({time.time()-t0:.0f}s)", flush=True)
        fc = hf.train_perkey_forecasts(train, val, test, cc, hf.stable_perkey_config(), int(ORIGIN))
        if not fc.empty:
            perkey.append(fc[["key", "year_week", "predicted"]])
    perkey = pd.concat(perkey, ignore_index=True) if perkey else pd.DataFrame(columns=["key", "year_week", "predicted"])

    # 3. combine: pooled deep for all, override own keys that got a per-key forecast
    pooled["key"] = pooled["key"].astype(str); pooled["year_week"] = pooled["year_week"].astype(int)
    perkey["key"] = perkey["key"].astype(str); perkey["year_week"] = perkey["year_week"].astype(int)
    got = set(perkey["key"])
    final = pd.concat([perkey, pooled[~pooled["key"].isin(got)]], ignore_index=True)
    final = final.groupby(["key", "year_week"], as_index=False)["predicted"].first()
    final.to_parquet("artifacts_th_local/final_forecast_router.parquet", index=False)
    print(f"[final] {len(got)} keys forecast by own model, {final['key'].nunique()-len(got)} by pooled deep", flush=True)

    # 4. eval vs LEGO
    p = lego_eval.attach_forecast(base, final, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    def ab(s):
        a = s.actual.values; f = s.ours.fillna(0).values; t = np.abs(a).sum(); return (1 - np.abs(f - a).sum() / t, (f - a).sum() / t)
    rows = []
    for c in sorted(p.Category.unique()):
        pc = p[p.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego))
        wb = "WIN" if (oa > la and abs(ob) < abs(lb)) else ("acc" if oa > la else ("bias" if abs(ob) < abs(lb) else "-"))
        rows.append((c, pc.actual.sum(), oa, la, ob, lb, wb))
        print(f"  {c:22s} acc {oa:+.3f}/{la:+.3f}  bias {ob:+.2f}/{lb:+.2f}  {wb}", flush=True)
    d = pd.DataFrame(rows, columns=["c", "v", "oa", "la", "ob", "lb", "wb"]); w = d.v / d.v.sum()
    print(f"  SALES-WTD acc {(d.oa*w).sum():.3f}/{(d.la*w).sum():.3f}  bias {(d.ob*w).sum():+.3f}/{(d.lb*w).sum():+.2f}"
          f" | acc-win {int((d.oa>d.la).sum())}/9, bias-win {int((d.ob.abs()<d.lb.abs()).sum())}/9, both {int(((d.oa>d.la)&(d.ob.abs()<d.lb.abs())).sum())}/9", flush=True)


if __name__ == "__main__":
    main()
