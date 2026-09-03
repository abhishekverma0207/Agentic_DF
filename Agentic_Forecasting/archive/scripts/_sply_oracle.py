"""Test whether adding a PURE seasonal-naive (SPLY = last-year-same-weeks actuals) base model raises
the per-key oracle on the hard 3 — it should nail the spiky-seasonal high-volume keys that every
trained model under-forecasts but LEGO captures. Also report the spiky keys' SPLY level."""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
HARD = ["FABRIC CLEANING", "HAIR CARE", "FOODS"]
CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]

# --- build pure-SPLY forecast for the test window (202603..202615 <- actual 202503..202515) ---
fr = []
for c in CATS:
    d = pd.read_parquet(f"sourcedata/THAILAND/by_category/{c}.parquet")
    d = d[d["key"].astype(str).str.endswith("_E1016")][["key", "year_week", "Actuals"]]
    fr.append(d)
hist = pd.concat(fr, ignore_index=True); hist["key"] = hist["key"].astype(str)
hist["year_week"] = pd.to_numeric(hist["year_week"], errors="coerce").astype("int64")
hist["Actuals"] = pd.to_numeric(hist["Actuals"], errors="coerce").fillna(0.0)
fut = list(range(202603, 202616))
rows = []
for w in fut:
    src = (w // 100 - 1) * 100 + (w % 100)  # w - 52 weeks (same ISO week last year)
    s = hist[hist.year_week == src][["key", "Actuals"]].rename(columns={"Actuals": "predicted"}); s["year_week"] = w
    rows.append(s)
sply = pd.concat(rows, ignore_index=True)[["key", "year_week", "predicted"]]
sply.to_parquet("artifacts_th_local/_sply_naive_fc.parquet", index=False)

MODELS = {}
for f in sorted(glob.glob("artifacts_th_local/deep_TFT_*_fc.parquet")):
    nm = f.split("deep_TFT_")[1].split("_fc")[0]
    if nm != "allkeys":
        d = pd.read_parquet(f); d["key"] = d["key"].astype(str); MODELS[nm] = d
for nm, path in [("perkey_router", "artifacts_th_local/final_forecast_router.parquet"),
                 ("seasonal", "artifacts_th_local/_seasonal_perkey_fc.parquet")]:
    d = pd.read_parquet(path); d["key"] = d["key"].astype(str); MODELS[nm] = d[["key", "year_week", "predicted"]]
WITH = dict(MODELS); WITH["SPLY"] = sply


def perkey_err(fc):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category.isin(HARD)].dropna(subset=["ours"])
    return p.groupby(["Category", "cs"]).apply(lambda d: pd.Series({"num": np.abs(d["ours"] - d["actual"]).sum(), "den": np.abs(d["actual"]).sum()}))


lego = base[base.Category.isin(HARD)].groupby(["Category", "cs"]).apply(
    lambda d: pd.Series({"lnum": np.abs(d["lego"] - d["actual"]).sum(), "den": np.abs(d["actual"]).sum()}))
idx = lego.index


def oracle(models):
    M = pd.DataFrame({"den": lego["den"], "lnum": lego["lnum"]}, index=idx)
    for nm, fc in models.items():
        M[nm] = perkey_err(fc)["num"].reindex(idx)
    M = M.reset_index()
    out = {}
    for cat in HARD:
        s = M[M.Category == cat]; cols = [c for c in models]
        orc = s[cols].min(axis=1)
        out[cat] = (1 - orc.sum() / s["den"].sum(), 1 - s["lnum"].sum() / s["den"].sum())
    return out


o0, o1 = oracle(MODELS), oracle(WITH)
print("=== oracle WITHOUT vs WITH pure-SPLY base model (13wk acc; LEGO in parens) ===")
for cat in HARD:
    print(f"  {cat:18s} oracle {o0[cat][0]:.3f} -> +SPLY {o1[cat][0]:.3f}  (LEGO {o0[cat][1]:.3f})")
# SPLY-only accuracy per category + spiky-key levels
print("=== SPLY-naive standalone acc + the 3 worst-under-forecast FABRIC keys ===")
sp = lego_eval.attach_forecast(base, sply, pred_col="predicted")
for cat in HARD:
    pc = sp[sp.Category == cat].dropna(subset=["ours"])
    a = 1 - np.abs(pc["ours"] - pc["actual"]).sum() / max(np.abs(pc["actual"]).sum(), 1e-9)
    print(f"  {cat:18s} SPLY-only acc {a:.3f}")
for cs in ["18851932464229", "18851932431511", "18851932431566"]:
    sub = base[base.cs == cs]; act = sub["actual"].sum(); leg = sub["lego"].sum()
    sv = sply[sply.key == cs + "_E1016"]["predicted"].sum()
    print(f"  {cs}: actual {act:.0f}  SPLY {sv:.0f} ({sv/act-1:+.0%})  LEGO {leg:.0f} ({leg/act-1:+.0%})")
