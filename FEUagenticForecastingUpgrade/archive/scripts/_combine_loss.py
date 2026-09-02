"""Combine the two deep models to get accurate AND centered: take the MAE TFT's per-week
SHAPE and scale it to the RMSE TFT's LEVEL. The level factor per category (or per key) is
derived purely from the two MODEL forecasts (sum RMSE / sum MAE) — no test actuals, so it's
a legitimate model-level bias correction. Eval bias + accuracy vs LEGO."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from utils import lego_eval

base = lego_eval.load_eval_base()
cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
mae = pd.read_parquet("artifacts_th_local/deep_TFT_MAE_fc.parquet").rename(columns={"predicted": "mae"})
rmse = pd.read_parquet("artifacts_th_local/deep_TFT_RMSE_fc.parquet").rename(columns={"predicted": "rmse"})
m = mae.merge(rmse, on=["key", "year_week"], how="inner")
m["cs"] = m["key"].astype(str).str[:-6]; m["Category"] = m["cs"].map(cs2cat)


def ev(fc, label):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    def ab(s):
        a = s["actual"].values; f = s["ours"].fillna(0).values; t = np.abs(a).sum()
        return (1 - np.abs(f - a).sum() / t, (f - a).sum() / t)
    rows = []
    for c in sorted(p.Category.unique()):
        pc = p[p.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego))
        rows.append((c, pc.actual.sum(), oa, ob, la, lb))
    d = pd.DataFrame(rows, columns=["c", "v", "oa", "ob", "la", "lb"]); w = d.v / d.v.sum()
    print(f"  {label:22s} acc {(d.oa*w).sum():.3f} (LEGO {(d.la*w).sum():.3f}, win {int((d.oa>d.la).sum())}/9) "
          f"| bias {(d.ob*w).sum():+.3f} (LEGO {(d.lb*w).sum():+.2f}, |b|-win {int((d.ob.abs()<d.lb.abs()).sum())}/9)")
    return d


def f(col):
    return m[["key", "year_week", col]].rename(columns={col: "predicted"})


print("baselines:")
ev(f("mae"), "MAE (shape)")
ev(f("rmse"), "RMSE (level)")
# per-CATEGORY level factor = sum(rmse)/sum(mae), applied to the MAE shape
catf = m.groupby("Category").apply(lambda d: d["rmse"].sum() / max(d["mae"].sum(), 1e-9))
m["mae_catlvl"] = m["mae"] * m["Category"].map(catf).fillna(1.0)
# per-KEY level factor (finer)
keyf = m.groupby("key").apply(lambda d: d["rmse"].sum() / max(d["mae"].sum(), 1e-9)).clip(0.5, 3.0)
m["mae_keylvl"] = m["mae"] * m["key"].map(keyf).fillna(1.0)
print("\nMAE shape scaled to RMSE level:")
ev(f("mae_catlvl"), "MAE-shape x cat-level")
ev(f("mae_keylvl"), "MAE-shape x key-level")
# also a few additive blends for reference
for wq in [0.5, 0.3]:
    m["b"] = wq * m["mae"] + (1 - wq) * m["rmse"]
    ev(f("b"), f"blend {wq:.1f}MAE+{1-wq:.1f}RMSE")
