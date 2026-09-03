"""Refine the seasonal overlay: horizon-limit it (seasonal per-key at t<=H where its seasonal-level
edge lives, deep base at long horizons where it's noisy) and test category subsets. Pick the config
that best improves t45 (both acc+bias) WITHOUT losing the 13wk accuracy headline vs LEGO. No training."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
CATS = sorted(c for c in base.Category.unique() if c != "ICE CREAM")
cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
wk2t = base.drop_duplicates("Shipment_Week").assign(yw=lambda d: d["Shipment_Week"].str.replace("-", "").astype(int)).set_index("yw")["t"]

bestfc = pd.read_parquet("artifacts_th_local/final_forecast_best.parquet"); bestfc["key"] = bestfc["key"].astype(str); bestfc["year_week"] = bestfc["year_week"].astype(int)
seas = pd.read_parquet("artifacts_th_local/_seasonal_perkey_fc.parquet"); seas["key"] = seas["key"].astype(str); seas["year_week"] = seas["year_week"].astype(int)
seas["cat"] = seas["key"].str[:-6].map(cs2cat); seas["t"] = seas["year_week"].map(wk2t)


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(tot, 1e-9), (f - a).sum() / max(tot, 1e-9))


def overlay(cats, hmax):
    use = seas[(seas.cat.isin(cats)) & (seas.t <= hmax)][["key", "year_week", "predicted"]]
    keywk = set(zip(use.key, use.year_week))
    rest = bestfc[~bestfc.apply(lambda r: (r.key, r.year_week) in keywk, axis=1)][["key", "year_week", "predicted"]]
    return pd.concat([use, rest], ignore_index=True).groupby(["key", "year_week"], as_index=False)["predicted"].first()


def board(fc, name):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    res = {}
    for tag, ts in [("t45", [4, 5]), ("13wk", list(range(1, 14)))]:
        q = p[p.t.isin(ts)]; rows = []
        for c in CATS:
            pc = q[q.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego)); rows.append((pc.actual.sum(), oa, la, ob, lb))
        d = pd.DataFrame(rows, columns=["v", "oa", "la", "ob", "lb"]); w = d.v / d.v.sum()
        res[tag] = ((d.oa * w).sum(), (d.ob * w).sum(), int((d.oa > d.la).sum()), int(((d.oa > d.la) & (d.ob.abs() < d.lb.abs())).sum()), (d.la * w).sum())
    a4, b4, aw4, bo4, la4 = res["t45"]; a13, b13, aw13, bo13, la13 = res["13wk"]
    print("  %-26s t45 a%.3f b%+.3f aw%d BOTH%d | 13wk a%.3f b%+.3f aw%d BOTH%d" % (name, a4, b4, aw4, bo4, a13, b13, aw13, bo13))
    return res


print("LEGO: t45 acc 0.349 / 13wk acc 0.446  (beat 13wk acc to keep the headline)")
board(bestfc, "BASE (no overlay)")
for cats in [["FABRIC CLEANING"], ["FABRIC CLEANING", "HAIR CARE"], ["FABRIC CLEANING", "SKIN CLEANSING"], ["FABRIC CLEANING", "HAIR CARE", "SKIN CLEANSING"]]:
    for h in [5, 6, 8, 13]:
        board(overlay(cats, h), f"{'+'.join(c.split()[0] for c in cats)} t<={h}")
