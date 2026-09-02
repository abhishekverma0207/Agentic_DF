"""Decision scorecard: compare LEGO vs bias-fixed pooled DEEP vs full ROUTER (per-key own at all
horizons) vs HROUTE (per-key own only at t<=HCUT, deep beyond). Per category at t+4/t+5 (primary)
and 13-week (secondary), sales-weighted. Picks the architecture that best beats LEGO at t45."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from utils import lego_eval

HCUT = int(sys.argv[1]) if len(sys.argv) > 1 else 6   # per-key for own keys at t<=HCUT

asg = pd.read_parquet("artifacts_th_local/_router_assignment.parquet"); asg["key"] = asg["key"].astype(str)
own_keys = set(asg.loc[asg.own == 1, "key"])
base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
rt = pd.read_parquet("artifacts_th_local/final_forecast_router.parquet"); rt["key"] = rt["key"].astype(str)

# bias-fixed pooled deep (Huber8 shape x RMSE cat-level)
H = pd.read_parquet("artifacts_th_local/deep_TFT_Huber8_fc.parquet").rename(columns={"predicted": "h"})
R = pd.read_parquet("artifacts_th_local/deep_TFT_RMSE_fc.parquet").rename(columns={"predicted": "r"})
mm = H.merge(R, on=["key", "year_week"]); mm["key"] = mm["key"].astype(str); mm["Category"] = mm["key"].str[:-6].map(cs2cat)
catf = mm.groupby("Category").apply(lambda d: d["r"].sum() / max(d["h"].sum(), 1e-9))
mm["predicted"] = (mm["h"] * mm["Category"].map(catf).fillna(1.0)).clip(lower=0)
deep = mm[["key", "year_week", "predicted"]].copy()

# year_week -> horizon t (from base mapping cs,Shipment_Week,t -> use the week ordering)
wk2t = base.drop_duplicates("Shipment_Week").assign(yw=lambda d: d["Shipment_Week"].str.replace("-", "").astype(int)).set_index("yw")["t"]
def add_t(d):
    d = d.copy(); d["t"] = d["year_week"].astype(int).map(wk2t); return d
rt, deep = add_t(rt), add_t(deep)

# HROUTE: per-key (from rt) for own keys at t<=HCUT, else deep
rt_own_short = rt[(rt.key.isin(own_keys)) & (rt.t <= HCUT)][["key", "year_week", "predicted"]]
keyset_short = set(zip(rt_own_short.key, rt_own_short.year_week))
deep_rest = deep[~deep.apply(lambda r: (r.key, r.year_week) in keyset_short, axis=1)][["key", "year_week", "predicted"]]
hroute = pd.concat([rt_own_short, deep_rest], ignore_index=True).groupby(["key", "year_week"], as_index=False)["predicted"].first()

CANDS = {"DEEP": deep[["key", "year_week", "predicted"]], "ROUTER": rt[["key", "year_week", "predicted"]], "HROUTE": hroute}


def ab(s):
    a = s.actual.values; f = s.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / tot, (f - a).sum() / tot)


def scorecard(name, fc, tsel):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[(p.Category != "ICE CREAM") & (p.t.isin(tsel))]
    rows = []
    for c in sorted(p.Category.unique()):
        pc = p[p.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego))
        rows.append((c, pc.actual.sum(), oa, la, ob, lb))
    d = pd.DataFrame(rows, columns=["c", "v", "oa", "la", "ob", "lb"]); w = d.v / d.v.sum()
    accw = (d.oa * w).sum(); legw = (d.la * w).sum(); bow = (d.ob * w).sum(); lbw = (d.lb * w).sum()
    aw = int((d.oa > d.la).sum()); bw = int((d.ob.abs() < d.lb.abs()).sum()); both = int(((d.oa > d.la) & (d.ob.abs() < d.lb.abs())).sum())
    print(f"  {name:8s} acc {accw:.3f} (LEGO {legw:.3f})  bias {bow:+.3f} (LEGO {lbw:+.3f})  | acc-win {aw}/9 bias-win {bw}/9 BOTH {both}/9")
    return d


print(f"HCUT={HCUT} (per-key for own keys at t<= this)\n=== t+4 & t+5 (PRIMARY) ===")
for n, fc in CANDS.items():
    scorecard(n, fc, [4, 5])
print("=== 13-week (secondary) ===")
for n, fc in CANDS.items():
    scorecard(n, fc, list(range(1, 14)))

# per-category t45 detail for the best (HROUTE)
print("=== HROUTE t45 per-category (acc ours/LEGO, bias ours/LEGO) ===")
p = lego_eval.attach_forecast(base, hroute, pred_col="predicted"); p = p[(p.Category != "ICE CREAM") & (p.t.isin([4, 5]))]
for c in sorted(p.Category.unique()):
    pc = p[p.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego))
    v = "WIN" if (oa > la and abs(ob) < abs(lb)) else ("acc" if oa > la else ("bias" if abs(ob) < abs(lb) else "-"))
    print(f"  {c:22s} acc {oa:+.3f}/{la:+.3f}  bias {ob:+.2f}/{lb:+.2f}  {v}")
