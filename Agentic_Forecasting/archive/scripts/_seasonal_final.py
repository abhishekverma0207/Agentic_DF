"""Assemble the FINAL deliverable: per-category selection base + FABRIC CLEANING seasonal per-key
overlay at horizons t<=6 (the Pareto-best config from _seasonal_refine). Backs up the pure base,
writes final_forecast_best.parquet (the canonical deliverable), prints the scorecard."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from utils import lego_eval

OVERLAY_CATS = ["FABRIC CLEANING"]
HMAX = 6
base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
CATS = sorted(c for c in base.Category.unique() if c != "ICE CREAM")
cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
wk2t = base.drop_duplicates("Shipment_Week").assign(yw=lambda d: d["Shipment_Week"].str.replace("-", "").astype(int)).set_index("yw")["t"]

bestfc = pd.read_parquet("artifacts_th_local/final_forecast_best.parquet"); bestfc["key"] = bestfc["key"].astype(str); bestfc["year_week"] = bestfc["year_week"].astype(int)
bestfc.to_parquet("artifacts_th_local/final_forecast_catselect_base.parquet", index=False)  # provenance backup
seas = pd.read_parquet("artifacts_th_local/_seasonal_perkey_fc.parquet"); seas["key"] = seas["key"].astype(str); seas["year_week"] = seas["year_week"].astype(int)
seas["cat"] = seas["key"].str[:-6].map(cs2cat); seas["t"] = seas["year_week"].map(wk2t)

use = seas[(seas.cat.isin(OVERLAY_CATS)) & (seas.t <= HMAX)][["key", "year_week", "predicted"]]
keywk = set(zip(use.key, use.year_week))
rest = bestfc[~bestfc.apply(lambda r: (r.key, r.year_week) in keywk, axis=1)][["key", "year_week", "predicted"]]
final = pd.concat([use, rest], ignore_index=True).groupby(["key", "year_week"], as_index=False)["predicted"].first()
final.to_parquet("artifacts_th_local/final_forecast_best.parquet", index=False)
print(f"overlaid {len(keywk)} (key,week) cells from {OVERLAY_CATS} at t<={HMAX}; final rows {len(final)}")


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(tot, 1e-9), (f - a).sum() / max(tot, 1e-9))


p = lego_eval.attach_forecast(base, final, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
for tag, ts in [("t45", [4, 5]), ("13wk", list(range(1, 14)))]:
    q = p[p.t.isin(ts)]; rows = []
    for c in CATS:
        pc = q[q.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego)); rows.append((pc.actual.sum(), oa, la, ob, lb))
    d = pd.DataFrame(rows, columns=["v", "oa", "la", "ob", "lb"]); w = d.v / d.v.sum()
    print("  FINAL %-4s acc %.3f (LEGO %.3f) bias %+.3f (LEGO %+.3f) | acc-win %d/9 bias-win %d/9 BOTH %d/9" % (
        tag, (d.oa * w).sum(), (d.la * w).sum(), (d.ob * w).sum(), (d.lb * w).sum(),
        int((d.oa > d.la).sum()), int((d.ob.abs() < d.lb.abs()).sum()), int(((d.oa > d.la) & (d.ob.abs() < d.lb.abs())).sum())))
