"""Explore GBM + deep ensembles for a BALANCED model (win t45 AND 13wk). Computes
per-horizon accuracy of GBM vs deep vs LEGO (where does each win?), then tests blends:
uniform weights, and a horizon-selective blend (deep where it beats GBM per horizon).
Usage: _ensemble.py [deep_model=NHITS]"""
import sys, glob; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from utils import lego_eval

DEEP = sys.argv[1] if len(sys.argv) > 1 else "NHITS"
base = lego_eval.load_eval_base()
gbm = pd.read_parquet("artifacts_th_local/final_forecast.parquet")[["key", "year_week", "predicted"]].rename(columns={"predicted": "g"})
deep = pd.read_parquet(f"artifacts_th_local/deep_{DEEP}_fc.parquet")[["key", "year_week", "predicted"]].rename(columns={"predicted": "d"})
m = gbm.merge(deep, on=["key", "year_week"], how="inner")
m["key"] = m["key"].astype(str); m["year_week"] = m["year_week"].astype(int)


def ev(fcol, label):
    fc = m[["key", "year_week", fcol]].rename(columns={fcol: "predicted"})
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    def acc(s, col):
        a = s["actual"].values; f = (s["ours"].fillna(0) if col == "ours" else s[col]).values
        t = np.abs(a).sum(); return 1 - np.abs(f - a).sum() / t if t else np.nan
    rows = []
    for c in sorted(p.Category.unique()):
        pc = p[p.Category == c]; p45 = pc[pc.t.isin([4, 5])]
        rows.append((c, pc.actual.sum(), acc(p45, "ours"), acc(p45, "lego"), acc(pc, "ours"), acc(pc, "lego")))
    d = pd.DataFrame(rows, columns=["c", "v", "o45", "l45", "o13", "l13"]); w = d.v / d.v.sum()
    print(f"  {label:28s} t45 {(d.o45*w).sum():.3f} (LEGO {(d.l45*w).sum():.3f}, win {int((d.o45>d.l45).sum())}/9) | "
          f"13wk {(d.o13*w).sum():.3f} (LEGO {(d.l13*w).sum():.3f}, win {int((d.o13>d.l13).sum())}/9)")
    return d


# horizon index t directly from week rank (202603->t1 .. 202615->t13)
weeks = sorted(m["year_week"].unique()); tmap = {w: i + 1 for i, w in enumerate(weeks)}
m["t"] = m["year_week"].map(tmap)

# per-horizon: where does each model win? (vol-wtd across cats, via separate attach)
def attach(fcol):
    return lego_eval.attach_forecast(base, m[["key", "year_week", fcol]].rename(columns={fcol: "predicted"}), pred_col="predicted")
pg, pd_ = attach("g"), attach("d")
ph = pg[["cs", "Shipment_Week", "Category", "t", "actual", "lego", "ours"]].rename(columns={"ours": "g"})
ph = ph.merge(pd_[["cs", "Shipment_Week", "ours"]].rename(columns={"ours": "d"}), on=["cs", "Shipment_Week"])
ph = ph[ph.Category != "ICE CREAM"]
print(f"per-horizon (vol-wtd acc): GBM  deep({DEEP})  LEGO")
for h in range(1, 14):
    phh = ph[ph.t == h]; a = phh.actual.values; t = np.abs(a).sum()
    if not t: continue
    g = 1 - np.abs(phh.g.fillna(0) - a).sum() / t
    dd = 1 - np.abs(phh.d.fillna(0) - a).sum() / t
    l = 1 - np.abs(phh.lego - a).sum() / t
    best = "GBM" if g >= max(dd, l) else ("deep" if dd >= l else "LEGO")
    print(f"  t+{h:<2d} {g:+.3f} {dd:+.3f} {l:+.3f}  best={best}")

print("\nblends (g=GBM weight):")
for wg in [1.0, 0.7, 0.5, 0.3, 0.0]:
    m["b"] = wg * m["g"] + (1 - wg) * m["d"]
    ev("b", f"blend g={wg:.1f}")
m["b"] = np.where(m["t"] >= 10, m["d"], m["g"]); ev("b", "horizon-sel (deep t>=10)")
m["b"] = np.where(m["t"].isin([1, 2, 3, 11, 12]), 0.5 * m["g"] + 0.5 * m["d"], m["g"]); ev("b", "blend GBM-bleed horizons")
