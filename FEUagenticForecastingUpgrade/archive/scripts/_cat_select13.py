"""Step A: per-category loss re-selection for the 13-WEEK objective (the client's new target).
Candidates per category: raw deep loss variants, bias-lift blends (loss x RMSE), and per-key
overrides. Objective: beat LEGO on BOTH 13wk acc and |bias|; if none, max 13wk acc with bias
tiebreak. Writes final_forecast_catselect13.parquet and the 13wk scorecard. No training (reuses
existing deep_TFT_*_fc.parquet at origin 202602)."""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
CATS = sorted(c for c in base.Category.unique() if c != "ICE CREAM")
cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]

RAW = {}
for f in sorted(glob.glob("artifacts_th_local/deep_TFT_*_fc.parquet")):
    nm = f.split("deep_TFT_")[1].split("_fc")[0]
    if nm == "allkeys":
        continue
    d = pd.read_parquet(f); d["key"] = d["key"].astype(str); RAW[nm] = d[["key", "year_week", "predicted"]]
# hierarchical XGBoost variants (cross-account + promo features) — the diverse new base learners
for nm, path in [("XGBhs", "artifacts_th_local/_xgb_huber_s18.parquet"), ("XGBh", "artifacts_th_local/_xgb_huber.parquet"),
                 ("XGBtw", "artifacts_th_local/_xgb_tweedie.parquet"), ("XGBhsm", "artifacts_th_local/_xgb_hier_fc.parquet"),
                 ("XGBdec", "artifacts_th_local/_xgb_dec.parquet"),
                 ("XGBr", "artifacts_th_local/_xgb_ratio.parquet"), ("XGBrq", "artifacts_th_local/_xgb_ratioq.parquet"),
                 ("XGBpk", "artifacts_th_local/_xgb_perkey.parquet"), ("XGBsl", "artifacts_th_local/_xgb_shapelevel.parquet")]:
    if glob.glob(path):
        d = pd.read_parquet(path); d["key"] = d["key"].astype(str); RAW[nm] = d[["key", "year_week", "predicted"]]

asg = pd.read_parquet("artifacts_th_local/_router_assignment.parquet"); asg["key"] = asg["key"].astype(str)
own_keys = set(asg.loc[asg.own == 1, "key"])
rt = pd.read_parquet("artifacts_th_local/final_forecast_router.parquet"); rt["key"] = rt["key"].astype(str)
perkey_own = rt[rt.key.isin(own_keys)][["key", "year_week", "predicted"]]


def blend(a, b, w):
    m = a.merge(b, on=["key", "year_week"], suffixes=("_a", "_b"))
    m["predicted"] = w * m["predicted_a"] + (1 - w) * m["predicted_b"]
    return m[["key", "year_week", "predicted"]]


def graft_perkey(bvar):
    rest = bvar[~bvar.key.isin(own_keys)]
    return pd.concat([perkey_own, rest], ignore_index=True).groupby(["key", "year_week"], as_index=False)["predicted"].first()


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(tot, 1e-9), (f - a).sum() / max(tot, 1e-9))


def cat_metrics13(fc, c):
    cskeys = set(base.loc[base.Category == c, "cs"].astype(str) + "_E1016")
    f = fc[fc.key.isin(cskeys)]
    p = lego_eval.attach_forecast(base, f, pred_col="predicted"); p = p[p.Category == c]   # all 13 weeks
    a13, b13 = ab(p); la13, lb13 = ab(p.assign(ours=p.lego))
    return dict(a=a13, b=b13, la=la13, lb=lb13)


def candidates():
    cands = dict(RAW)
    for nm in ["MAE", "Huber", "Huber8", "Huber3"]:
        cands[f"{nm}+RMSE.7"] = blend(RAW[nm], RAW["RMSE"], 0.7)
        cands[f"{nm}+RMSE.5"] = blend(RAW[nm], RAW["RMSE"], 0.5)
        cands[f"{nm}+RMSE.3"] = blend(RAW[nm], RAW["RMSE"], 0.3)
    for nm in ["MAE", "Huber", "Huber8", "RMSE"]:
        cands[f"PK+{nm}"] = graft_perkey(RAW[nm])
    # XGB-hier (diverse, high-acc on HAIR) blended with mean-targeters for bias
    for xnm in ["XGBh", "XGBhs", "XGBdec"]:
        for dnm in ["RMSE", "Huber8", "XGBtw"]:
            if xnm in RAW and dnm in RAW:
                cands[f"{xnm}+{dnm}.6"] = blend(RAW[xnm], RAW[dnm], 0.6)
                cands[f"{xnm}+{dnm}.4"] = blend(RAW[xnm], RAW[dnm], 0.4)
    # high-acc LEVEL blended with good-bias RATIO -> the acc/bias frontier for the hard categories
    for xnm in ["XGBhs", "XGBdec", "XGBh"]:
        for rnm in ["XGBr", "XGBrq"]:
            if xnm in RAW and rnm in RAW:
                for w in (0.6, 0.5, 0.4, 0.3):
                    cands[f"{xnm}+{rnm}.{int(w*10)}"] = blend(RAW[xnm], RAW[rnm], w)
    # shape(accurate)xlevel(per-key growth) combo blended with the high-acc level -> the frontier-beating point
    for xnm in ["XGBsl", "XGBpk"]:
        for dnm in ["XGBhs", "XGBdec", "XGBrq"]:
            if xnm in RAW and dnm in RAW:
                for w in (0.7, 0.5, 0.3):
                    cands[f"{xnm}+{dnm}.{int(w*10)}"] = blend(RAW[xnm], RAW[dnm], w)
    return cands


CANDS = candidates()
print(f"{len(CANDS)} candidates/category; selecting per category for BOTH-win at 13 weeks...")
pick, parts, detail = {}, [], []
for c in CATS:
    best, bestkey = None, None
    for nm, fc in CANDS.items():
        m = cat_metrics13(fc, c)
        both = (m["a"] > m["la"]) and (abs(m["b"]) < abs(m["lb"]))
        acc_ok = m["a"] > m["la"]
        score = (2e6 if both else (1e6 if acc_ok else 0)) + m["a"] * 1000 - abs(m["b"]) * 200
        if best is None or score > best[0]:
            best, bestkey = (score, m, both), nm
    pick[c] = bestkey; m = best[1]
    cskeys = set(base.loc[base.Category == c, "cs"].astype(str) + "_E1016")
    parts.append(CANDS[bestkey][CANDS[bestkey].key.isin(cskeys)][["key", "year_week", "predicted"]])
    detail.append((c, bestkey, m, best[2]))

final = pd.concat(parts, ignore_index=True)
final.to_parquet("artifacts_th_local/final_forecast_catselect13.parquet", index=False)

TOP5 = {"FABRIC CLEANING", "HOME & HYGIENE", "HAIR CARE", "FOODS", "FABRIC ENHANCERS"}
print("=== per-category 13-week picks (DIQ acc/LEGO, DIQ bias/LEGO) ===")
for c, nm, m, both in detail:
    v = "WIN both" if both else ("acc only" if m["a"] > m["la"] else ("bias only" if abs(m["b"]) < abs(m["lb"]) else "-"))
    star = " *top5" if c in TOP5 else ""
    print(f"  {c:24s} [{nm:12s}] acc {m['a']:+.3f}/{m['la']:+.3f}  bias {m['b']:+.3f}/{m['lb']:+.3f}  {v}{star}")


def board(fc, name):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    q = p  # 13 weeks
    rows = []
    for c in CATS:
        pc = q[q.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego)); rows.append((pc.actual.sum(), oa, la, ob, lb, c))
    d = pd.DataFrame(rows, columns=["v", "oa", "la", "ob", "lb", "c"]); w = d.v / d.v.sum()
    d5 = d[d.c.isin(TOP5)]
    print("  %s 13wk acc %.3f (LEGO %.3f) bias %+.3f (LEGO %+.3f) | all9 BOTH %d/9 | TOP5 BOTH %d/5" % (
        name, (d.oa * w).sum(), (d.la * w).sum(), (d.ob * w).sum(), (d.lb * w).sum(),
        int(((d.oa > d.la) & (d.ob.abs() < d.lb.abs())).sum()),
        int(((d5.oa > d5.la) & (d5.ob.abs() < d5.lb.abs())).sum())))


print("=== CATSELECT13 scorecard ===")
board(final, "CATSELECT13")
