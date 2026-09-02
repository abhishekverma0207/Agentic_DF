"""Find the best bias-accuracy point by combining the accurate-shape model (Huber8) with
the centered-level model (RMSE): additive blends + shape-scaled-to-level. All free (saved
forecasts). Report accuracy & bias vs LEGO; flag points that beat LEGO bias at high accuracy."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from utils import lego_eval

base = lego_eval.load_eval_base()
cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
H = pd.read_parquet("artifacts_th_local/deep_TFT_Huber8_fc.parquet").rename(columns={"predicted": "h"})
R = pd.read_parquet("artifacts_th_local/deep_TFT_RMSE_fc.parquet").rename(columns={"predicted": "r"})
m = H.merge(R, on=["key", "year_week"]); m["cs"] = m["key"].astype(str).str[:-6]; m["Category"] = m["cs"].map(cs2cat)
LEGO_ACC, LEGO_BIAS = 0.446, -0.06


def ev(col, label):
    fc = m[["key", "year_week", col]].rename(columns={col: "predicted"})
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    def ab(s):
        a = s["actual"].values; f = s["ours"].fillna(0).values; t = np.abs(a).sum()
        return (1 - np.abs(f - a).sum() / t, (f - a).sum() / t)
    d = pd.DataFrame([(c, p[p.Category == c].actual.sum(), *ab(p[p.Category == c])) for c in p.Category.unique()],
                     columns=["c", "v", "oa", "ob"]); w = d.v / d.v.sum()
    acc, bias = (d.oa * w).sum(), (d.ob * w).sum()
    flag = " <== beats LEGO bias" + ("" if acc >= LEGO_ACC else f", acc -{LEGO_ACC-acc:.3f}") if abs(bias) < abs(LEGO_BIAS) else ""
    print(f"  {label:26s} acc {acc:.3f}  bias {bias:+.3f}{flag}")
    return acc, bias


print(f"target: LEGO acc {LEGO_ACC} / bias {LEGO_BIAS}\n")
ev("h", "Huber8 (shape)"); ev("r", "RMSE (level)")
print("  -- additive blends (variance reduction) --")
for wq in [0.6, 0.5, 0.4, 0.3, 0.2]:
    m["b"] = wq * m["h"] + (1 - wq) * m["r"]; ev("b", f"{wq:.1f}Huber8+{1-wq:.1f}RMSE")
print("  -- Huber8 shape scaled to RMSE level --")
catf = m.groupby("Category").apply(lambda d: d["r"].sum() / max(d["h"].sum(), 1e-9))
m["hcl"] = m["h"] * m["Category"].map(catf).fillna(1.0); ev("hcl", "Huber8 x cat-level")
keyf = m.groupby("key").apply(lambda d: d["r"].sum() / max(d["h"].sum(), 1e-9)).clip(0.5, 3.0)
m["hkl"] = m["h"] * m["key"].map(keyf).fillna(1.0); ev("hkl", "Huber8 x key-level")
# half-way level scale (mild)
m["hcl5"] = m["h"] * (0.5 + 0.5 * m["Category"].map(catf).fillna(1.0)); ev("hcl5", "Huber8 x half-cat-level")
