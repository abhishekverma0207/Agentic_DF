"""Per-category model-selection decision matrix. For each category, score every raw deep loss
variant at t45 and 13wk vs LEGO, then pick the per-category winner that beats LEGO on BOTH acc and
|bias| at t45 (fallback: best acc with bias within tolerance). Reports the selected ensemble's
scorecard and writes final_forecast_catselect.parquet."""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
CATS = sorted(c for c in base.Category.unique() if c != "ICE CREAM")
VARIANTS = {}
for f in sorted(glob.glob("artifacts_th_local/deep_TFT_*_fc.parquet")):
    nm = f.split("deep_TFT_")[1].split("_fc")[0]
    if nm == "allkeys":
        continue
    d = pd.read_parquet(f); d["key"] = d["key"].astype(str); VARIANTS[nm] = d


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / tot, (f - a).sum() / tot)


# precompute per-(variant, category) acc/bias at t45 and 13wk
def cat_scores(fc):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    out = {}
    for c in CATS:
        pc = p[p.Category == c]; pc45 = pc[pc.t.isin([4, 5])]
        if len(pc) == 0:
            continue
        out[c] = dict(a45=ab(pc45)[0], b45=ab(pc45)[1], a13=ab(pc)[0], b13=ab(pc)[1],
                      la45=ab(pc45.assign(ours=pc45.lego))[0], lb45=ab(pc45.assign(ours=pc45.lego))[1],
                      la13=ab(pc.assign(ours=pc.lego))[0], lb13=ab(pc.assign(ours=pc.lego))[1], v=pc.actual.sum())
    return out


SC = {nm: cat_scores(fc) for nm, fc in VARIANTS.items()}

# selection: per category, choose variant maximizing a t45-weighted objective that rewards beating LEGO
# on both. score = (a45 - la45) + 0.5*(a13 - la13) + 0.5*(|lb45|-|b45|)  -> bigger = better vs LEGO
print("per-category t45 acc by variant (LEGO in []):")
hdr = "  %-22s " % "cat" + " ".join("%7s" % v for v in VARIANTS) + "   LEGO45  pick"
print(hdr)
pick = {}
for c in CATS:
    best, bestscore = None, -1e9
    cells = []
    for nm in VARIANTS:
        s = SC[nm].get(c)
        if not s:
            cells.append("  n/a "); continue
        cells.append("%7.3f" % s["a45"])
        obj = (s["a45"] - s["la45"]) + 0.5 * (s["a13"] - s["la13"]) + 0.4 * (abs(s["lb45"]) - abs(s["b45"]))
        if obj > bestscore:
            bestscore, best = obj, nm
    pick[c] = best
    la45 = SC[list(VARIANTS)[0]][c]["la45"] if c in SC[list(VARIANTS)[0]] else float("nan")
    print("  %-22s %s   %6.3f  %s" % (c, " ".join(cells), la45, best))

# assemble per-category selected forecast
parts = []
for c in CATS:
    cskeys = set(base.loc[base.Category == c, "cs"].astype(str) + "_E1016")
    d = VARIANTS[pick[c]]; parts.append(d[d.key.isin(cskeys)][["key", "year_week", "predicted"]])
sel = pd.concat(parts, ignore_index=True)
sel.to_parquet("artifacts_th_local/final_forecast_catselect.parquet", index=False)


def board(fc, name):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    for tag, tsel in [("t45", [4, 5]), ("13wk", list(range(1, 14)))]:
        q = p[p.t.isin(tsel)]; rows = []
        for c in CATS:
            pc = q[q.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego)); rows.append((pc.actual.sum(), oa, la, ob, lb))
        d = pd.DataFrame(rows, columns=["v", "oa", "la", "ob", "lb"]); w = d.v / d.v.sum()
        print("  %s %-4s acc %.3f (LEGO %.3f) bias %+.3f (LEGO %+.3f) | acc-win %d/9 bias-win %d/9 BOTH %d/9" % (
            name, tag, (d.oa * w).sum(), (d.la * w).sum(), (d.ob * w).sum(), (d.lb * w).sum(),
            int((d.oa > d.la).sum()), int((d.ob.abs() < d.lb.abs()).sum()), int(((d.oa > d.la) & (d.ob.abs() < d.lb.abs())).sum())))


print("=== CAT-SELECT scorecard ===")
board(sel, "CATSEL")
print("=== per-category CAT-SELECT t45 detail ===")
p = lego_eval.attach_forecast(base, sel, pred_col="predicted"); p = p[(p.Category != "ICE CREAM") & p.t.isin([4, 5])]
for c in CATS:
    pc = p[p.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego))
    v = "WIN" if (oa > la and abs(ob) < abs(lb)) else ("acc" if oa > la else ("bias" if abs(ob) < abs(lb) else "-"))
    print("  %-22s [%s] acc %+.3f/%+.3f bias %+.2f/%+.2f %s" % (c, pick[c], oa, la, ob, lb, v))
