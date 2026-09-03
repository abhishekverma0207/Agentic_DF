"""Size a REALISTIC stacker: instead of per-key oracle (upper bound), segment keys by forecastability
(volume x spikiness) and pick the best model PER SEGMENT (robust — many keys per segment). Reports
per-category 13wk accuracy vs LEGO and the segment->model map. If a coarse segment-stacker comfortably
beats LEGO on the test window, the leak-free 202541-learned version is worth building."""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
HARD = ["FABRIC CLEANING", "HAIR CARE", "FOODS"]
F = pd.read_parquet("artifacts_th_local/_key_features.parquet"); F["cs"] = F["key"].astype(str).str[:-6]
F = F.set_index("cs")

MODELS = {}
for f in sorted(glob.glob("artifacts_th_local/deep_TFT_*_fc.parquet")):
    nm = f.split("deep_TFT_")[1].split("_fc")[0]
    if nm != "allkeys":
        d = pd.read_parquet(f); d["key"] = d["key"].astype(str); MODELS[nm] = d
for nm, path in [("perkey_router", "artifacts_th_local/final_forecast_router.parquet"),
                 ("seasonal", "artifacts_th_local/_seasonal_perkey_fc.parquet")]:
    d = pd.read_parquet(path); d["key"] = d["key"].astype(str); MODELS[nm] = d[["key", "year_week", "predicted"]]


def perkey_err(fc):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category.isin(HARD)].dropna(subset=["ours"])
    return p.groupby(["Category", "cs"]).apply(lambda d: pd.Series({
        "num": np.abs(d["ours"] - d["actual"]).sum(), "den": np.abs(d["actual"]).sum()}))


E = {}
for nm, fc in MODELS.items():
    E[nm] = perkey_err(fc)
lego = base[base.Category.isin(HARD)].groupby(["Category", "cs"]).apply(
    lambda d: pd.Series({"lnum": np.abs(d["lego"] - d["actual"]).sum(), "den": np.abs(d["actual"]).sum()}))

idx = lego.index
df = pd.DataFrame({"den": lego["den"], "lnum": lego["lnum"]}, index=idx)
for nm in MODELS:
    df[f"n_{nm}"] = E[nm]["num"].reindex(idx)
df = df.reset_index()
df["cv2"] = df["cs"].map(F["CV2"]); df["vol"] = df["cs"].map(F["vol"])
df["spike"] = df["cs"].map(F["spike"])

OUR = list(MODELS)
print("=== per-segment stacker (segment = vol-tertile x CV2-tertile within category) vs LEGO 13wk ===")
for cat in HARD:
    s = df[df.Category == cat].copy()
    s["vt"] = pd.qcut(s["vol"].rank(method="first"), 3, labels=["loV", "midV", "hiV"])
    s["ct"] = pd.qcut(s["cv2"].rank(method="first"), 3, labels=["smooth", "mid", "spiky"])
    s["seg"] = s["vt"].astype(str) + "/" + s["ct"].astype(str)
    # pick best model per segment by total WAPE on that segment
    picks, stack_num = {}, np.zeros(len(s))
    for seg, g in s.groupby("seg"):
        accs = {nm: 1 - g[f"n_{nm}"].sum() / g["den"].sum() for nm in OUR}
        best = max(accs, key=accs.get); picks[seg] = (best, accs[best])
    s["pick"] = s["seg"].map(lambda x: picks[x][0])
    stack_num = s.apply(lambda r: r[f"n_{r['pick']}"], axis=1)
    accS = 1 - stack_num.sum() / s["den"].sum(); accL = 1 - s["lnum"].sum() / s["den"].sum()
    print(f"\n  {cat}: SEG-STACK acc {accS:.3f} vs LEGO {accL:.3f}  ({'BEATS' if accS > accL else 'LOSES'} by {accS-accL:+.3f})")
    for seg in sorted(picks):
        nseg = (s.seg == seg).sum(); print(f"    {seg:14s} -> {picks[seg][0]:14s} (seg acc {picks[seg][1]:.3f}, {nseg} keys)")
