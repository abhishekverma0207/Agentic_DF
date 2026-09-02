"""Surgical bias fix: the FABRIC bias loss is concentrated in the top high-volume keys we under-forecast.
Keep the accurate GLOBAL weekly shape, but rescale ONLY the top-N highest-volume keys' total to the
per-key model level (which captures the growth). Leak-free: top-N by PRE-ORIGIN history volume; per-key
level is as-of origin. Sweep N to see if FABRIC/HAIR cross to both-win."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
glob = pd.read_parquet("artifacts_th_local/final_forecast_catselect13.parquet"); glob["key"] = glob["key"].astype(str); glob["year_week"] = glob["year_week"].astype(int)
pk = pd.read_parquet("artifacts_th_local/_xgb_perkey.parquet"); pk["key"] = pk["key"].astype(str)
F = pd.read_parquet("artifacts_th_local/_key_features.parquet"); F["key"] = F["key"].astype(str)  # history vol (leak-free)
volmap = F.set_index("key")["vol"]


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; t = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(t, 1e-9), (f - a).sum() / max(t, 1e-9))


def cat_keys(cat):
    return set(base.loc[base.Category == cat, "cs"].astype(str) + "_E1016")


for cat in ["FABRIC CLEANING", "HAIR CARE"]:
    ck = cat_keys(cat)
    pkt = pk[pk.key.isin(ck)].groupby("key")["predicted"].sum()       # per-key model total (growth level)
    glt = glob[glob.key.isin(ck)].groupby("key")["predicted"].sum()    # current global total
    ranked = sorted([k for k in ck if k in volmap.index], key=lambda k: -volmap.get(k, 0))
    base0 = ab(lego_eval.attach_forecast(base, glob[glob.key.isin(ck)], pred_col="predicted").pipe(lambda p: p[p.Category == cat]))
    la, lb = ab(lego_eval.attach_forecast(base, glob[glob.key.isin(ck)], pred_col="predicted").pipe(lambda p: p[p.Category == cat]).assign(ours=lambda d: d.lego))
    print(f"\n=== {cat}: base acc {base0[0]:+.3f}/{la:+.3f} bias {base0[1]:+.3f}/{lb:+.3f} ===")
    for N in [5, 10, 20, 40, 80]:
        lift = set(ranked[:N])
        fac = (pkt / glt.replace(0, np.nan)).clip(0.7, 3.0)
        g = glob.copy()
        m = g.key.isin(lift)
        g.loc[m, "predicted"] = g.loc[m, "predicted"] * g.loc[m, "key"].map(fac).fillna(1.0)
        p = lego_eval.attach_forecast(base, g[g.key.isin(ck)], pred_col="predicted"); p = p[p.Category == cat]
        a, b = ab(p); v = "WIN both" if (a > la and abs(b) < abs(lb)) else ("acc" if a > la else ("bias" if abs(b) < abs(lb) else "-"))
        print(f"  lift top-{N:>2d} hv keys -> per-key level:  acc {a:+.3f}  bias {b:+.3f}   {v}")
