"""Final per-category selector. Candidates per category: raw deep loss variants, bias-lift blends
(acc-best x RMSE), and per-key override (own keys -> per-key DMH, rest -> deep). Objective: pick the
candidate that beats LEGO on BOTH t45 acc and |bias|; if none, max t45 acc with bias as tiebreak.
Honours the user's philosophy (per-key where it wins) without forcing it where it loses.
Writes final_forecast_best.parquet and prints the scorecard."""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
CATS = sorted(c for c in base.Category.unique() if c != "ICE CREAM")
cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]

# raw deep loss variants
RAW = {}
for f in sorted(glob.glob("artifacts_th_local/deep_TFT_*_fc.parquet")):
    nm = f.split("deep_TFT_")[1].split("_fc")[0]
    if nm == "allkeys":
        continue
    d = pd.read_parquet(f); d["key"] = d["key"].astype(str); RAW[nm] = d[["key", "year_week", "predicted"]]

# per-key own-key forecasts (from the router final): own keys carry per-key DMH; we graft onto any base
asg = pd.read_parquet("artifacts_th_local/_router_assignment.parquet"); asg["key"] = asg["key"].astype(str)
own_keys = set(asg.loc[asg.own == 1, "key"])
rt = pd.read_parquet("artifacts_th_local/final_forecast_router.parquet"); rt["key"] = rt["key"].astype(str)
perkey_own = rt[rt.key.isin(own_keys)][["key", "year_week", "predicted"]]


def blend(a, b, w):
    m = a.merge(b, on=["key", "year_week"], suffixes=("_a", "_b"))
    m["predicted"] = w * m["predicted_a"] + (1 - w) * m["predicted_b"]
    return m[["key", "year_week", "predicted"]]


def graft_perkey(bvar):
    """own keys -> per-key DMH, everyone else -> base variant."""
    rest = bvar[~bvar.key.isin(own_keys)]
    return pd.concat([perkey_own, rest], ignore_index=True).groupby(["key", "year_week"], as_index=False)["predicted"].first()


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(tot, 1e-9), (f - a).sum() / max(tot, 1e-9))


def cat_metrics(fc, c):
    cskeys = set(base.loc[base.Category == c, "cs"].astype(str) + "_E1016")
    f = fc[fc.key.isin(cskeys)]
    p = lego_eval.attach_forecast(base, f, pred_col="predicted"); p = p[(p.Category == c)]
    p45 = p[p.t.isin([4, 5])]
    a45, b45 = ab(p45); la45, lb45 = ab(p45.assign(ours=p45.lego))
    a13, b13 = ab(p); la13, lb13 = ab(p.assign(ours=p.lego))
    return dict(a45=a45, b45=b45, la45=la45, lb45=lb45, a13=a13, b13=b13, la13=la13, lb13=lb13)


# candidate set per category
def candidates():
    cands = dict(RAW)
    accbest = "Huber8"  # mid default for blends; per-cat we blend each raw with RMSE
    for nm in ["MAE", "Huber", "Huber8", "Huber3"]:
        cands[f"{nm}+RMSE.7"] = blend(RAW[nm], RAW["RMSE"], 0.7)
        cands[f"{nm}+RMSE.5"] = blend(RAW[nm], RAW["RMSE"], 0.5)
    for nm in ["MAE", "Huber", "Huber8", "RMSE"]:
        cands[f"PK+{nm}"] = graft_perkey(RAW[nm])
    return cands


CANDS = candidates()
print(f"{len(CANDS)} candidates/category; selecting per category for BOTH-win at t45...")
pick, parts, detail = {}, [], []
for c in CATS:
    best, bestkey = None, None
    for nm, fc in CANDS.items():
        m = cat_metrics(fc, c)
        both = (m["a45"] > m["la45"]) and (abs(m["b45"]) < abs(m["lb45"]))
        acc_ok = m["a45"] > m["la45"]
        # objective: both-win >> acc-win >> raw acc, with small bias preference
        score = (2e6 if both else (1e6 if acc_ok else 0)) + m["a45"] * 1000 - abs(m["b45"]) * 200
        if best is None or score > best[0]:
            best, bestkey = (score, m, both), nm
    pick[c] = bestkey; m = best[1]
    cskeys = set(base.loc[base.Category == c, "cs"].astype(str) + "_E1016")
    parts.append(CANDS[bestkey][CANDS[bestkey].key.isin(cskeys)][["key", "year_week", "predicted"]])
    v = "WIN" if best[2] else ("acc" if m["a45"] > m["la45"] else "-")
    detail.append((c, bestkey, m, v))

final = pd.concat(parts, ignore_index=True)
final.to_parquet("artifacts_th_local/final_forecast_best.parquet", index=False)


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


print("=== per-category picks (t45) ===")
for c, nm, m, v in detail:
    print("  %-22s [%-12s] acc %+.3f/%+.3f bias %+.2f/%+.2f %s" % (c, nm, m["a45"], m["la45"], m["b45"], m["lb45"], v))
print("=== FINAL_BEST scorecard ===")
board(final, "BEST")
