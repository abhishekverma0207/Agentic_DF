"""Seasonal-trust bias fix. The FABRIC bias loss is driven by high-volume keys where the Q1 seasonal
level (SPLY = same weeks last year) holds, but recent demand dipped so we under-forecast (e.g. 431511:
SPLY 83k ~= actual 79k, but recent 22k -> we forecast 56k). Fix: for keys where SPLY has PROVEN
block-reliability on past years, lift the forecast level UPWARD toward SPLY (never down -> can't hurt
the genuine-growth keys like 464229 where DIQ>SPLY). Keep the accurate weekly shape. Fully leak-free:
SPLY, reliability, recent are all pre-origin. Sweeps strength; reports per category + all-9 vs LEGO."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]
TOP5 = {"FABRIC CLEANING", "HOME & HYGIENE", "HAIR CARE", "FOODS", "FABRIC ENHANCERS"}
glob = pd.read_parquet("artifacts_th_local/final_forecast_catselect13.parquet"); glob["key"] = glob["key"].astype(str); glob["year_week"] = glob["year_week"].astype(int)

# E1016 history -> SPLY, block reliability, recent
fr = []
for c in CATS:
    d = pd.read_parquet(f"sourcedata/THAILAND/by_category/{c}.parquet", columns=["key", "year_week", "Actuals"])
    d = d[d["key"].astype(str).str.endswith("_E1016")]; fr.append(d)
h = pd.concat(fr, ignore_index=True); h["key"] = h["key"].astype(str)
h["year_week"] = pd.to_numeric(h["year_week"], errors="coerce").astype("int64")
h["Actuals"] = pd.to_numeric(h["Actuals"], errors="coerce").fillna(0.0)


def blk(lo, hi):
    return h[(h.year_week >= lo) & (h.year_week <= hi)].groupby("key")["Actuals"].sum()


sply_fut = blk(202503, 202515)                       # same weeks last year = level anchor for 202603..15
a25, s25 = blk(202503, 202515), blk(202403, 202415)  # 2025 block vs its SPLY -> reliability probe (leak-free)
a24, s24 = blk(202403, 202415), blk(202303, 202315)
rel = pd.DataFrame({"a25": a25, "s25": s25, "a24": a24, "s24": s24})
rel["e25"] = (rel.a25 - rel.s25).abs() / rel.a25.replace(0, np.nan)
rel["e24"] = (rel.a24 - rel.s24).abs() / rel.a24.replace(0, np.nan)
rel["score"] = (1 - rel[["e25", "e24"]].mean(axis=1)).clip(0, 1)   # high = SPLY reliably predicts the block
relmap = rel["score"].fillna(0.3)                                  # modest default when history is short
recent = blk(202549, 202602) / 6 * 13                             # recent run-rate over the block


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; t = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(t, 1e-9), (f - a).sum() / max(t, 1e-9))


def board(fc, name):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    rows = [(c, p[p.Category == c].actual.abs().sum(), *ab(p[p.Category == c]), *ab(p[p.Category == c].assign(ours=p[p.Category == c].lego))) for c in sorted(p.Category.unique())]
    d = pd.DataFrame(rows, columns=["c", "v", "oa", "ob", "la", "lb"])
    for grp, nm in [(d, "all9"), (d[d.c.isin(TOP5)], "top5")]:
        w = grp.v / grp.v.sum()
        print(f"  {name} {nm}: acc {(grp.oa*w).sum():.3f}/{(grp.la*w).sum():.3f} bias {(grp.ob*w).sum():+.3f}/{(grp.lb*w).sum():+.3f} | acc-win {int((grp.oa>grp.la).sum())} both {int(((grp.oa>grp.la)&(grp.ob.abs()<grp.lb.abs())).sum())}")
    return d


gt = glob.groupby("key")["predicted"].sum()
print("BASE (no fix):"); board(glob, "base")
for STRENGTH, RELMIN, VOLMIN in [(1.0, 0.5, 1.0), (1.0, 0.6, 1.0), (0.75, 0.5, 1.0)]:
    target = {}
    for k in gt.index:
        g0 = gt[k]; sp = sply_fut.get(k, 0.0); rl = relmap.get(k, 0.3); rc = recent.get(k, 0.0)
        # lift only UP, only reliable-SPLY keys, only where recent dipped below SPLY (the 431511 pattern)
        if rl >= RELMIN and sp > g0 and sp > rc and g0 > VOLMIN:
            target[k] = g0 + STRENGTH * rl * (sp - g0)
        else:
            target[k] = g0
    fac = pd.Series(target) / gt.replace(0, np.nan); fac = fac.clip(1.0, 3.0).fillna(1.0)
    g = glob.copy(); g["predicted"] = g["predicted"] * g["key"].map(fac).fillna(1.0)
    print(f"--- seasonal-trust STRENGTH={STRENGTH} RELMIN={RELMIN} ({int((fac>1.01).sum())} keys lifted) ---")
    board(g, "trust")
    g.to_parquet(f"artifacts_th_local/_seasonal_trust_{int(STRENGTH*100)}_{int(RELMIN*100)}.parquet", index=False)
