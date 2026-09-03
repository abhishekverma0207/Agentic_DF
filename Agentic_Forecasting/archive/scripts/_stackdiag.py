"""DEEP DIAGNOSTIC: is stacking viable? For each hard category, compute per-key 13wk WAPE of every
model we have + LEGO, then the per-key ORACLE (best of our models per key). If the oracle beats
LEGO, a learned stacker can win -> build it. If even the oracle loses, our base learners are all
weaker -> we need a new LEGO-class base model. Also: per-key win-rate, horizon decomposition, and
top-volume-key error structure (level vs timing)."""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
HARD = ["FABRIC CLEANING", "HAIR CARE", "FOODS"]

MODELS = {}
for f in sorted(glob.glob("artifacts_th_local/deep_TFT_*_fc.parquet")):
    nm = f.split("deep_TFT_")[1].split("_fc")[0]
    if nm != "allkeys":
        d = pd.read_parquet(f); d["key"] = d["key"].astype(str); MODELS[nm] = d
for nm, path in [("perkey_router", "artifacts_th_local/final_forecast_router.parquet"),
                 ("seasonal", "artifacts_th_local/_seasonal_perkey_fc.parquet"),
                 ("catselect13", "artifacts_th_local/final_forecast_catselect13.parquet")]:
    if glob.glob(path):
        d = pd.read_parquet(path); d["key"] = d["key"].astype(str); MODELS[nm] = d[["key", "year_week", "predicted"]]


def perkey_wape(fc):
    """return Series key->WAPE over 13 weeks (cs grain)."""
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted")
    p = p[p.Category.isin(HARD)].dropna(subset=["ours"])
    g = p.groupby(["Category", "cs"]).apply(lambda d: pd.Series({
        "w": np.abs(d["ours"] - d["actual"]).sum() / max(np.abs(d["actual"]).sum(), 1e-9),
        "vol": np.abs(d["actual"]).sum()}))
    return g


# LEGO per-key WAPE + actual volume
pl = lego_eval.attach_forecast(base, MODELS["Huber8"], pred_col="predicted")  # just to get the frame shape
pl = base[base.Category.isin(HARD)].copy()
legg = pl.groupby(["Category", "cs"]).apply(lambda d: pd.Series({
    "wL": np.abs(d["lego"] - d["actual"]).sum() / max(np.abs(d["actual"]).sum(), 1e-9),
    "vol": np.abs(d["actual"]).sum()}))

# per-model per-key WAPE matrix
W = pd.DataFrame({"vol": legg["vol"], "LEGO": legg["wL"]})
for nm, fc in MODELS.items():
    pk = perkey_wape(fc)
    W[nm] = pk["w"].reindex(W.index)
W = W.reset_index()

ourcols = [c for c in MODELS]
print("=== per-key 13wk: ORACLE (best of our models per key) vs LEGO, volume-weighted accuracy ===")
for cat in HARD:
    sub = W[W.Category == cat].copy()
    sub["oracle"] = sub[ourcols].min(axis=1)  # best (lowest WAPE) of our models per key
    vw = sub["vol"] / sub["vol"].sum()
    # volume-weighted mean WAPE -> accuracy = 1 - that
    accL = 1 - (sub["LEGO"] * vw).sum(); accO = 1 - (sub["oracle"] * vw).sum()
    # best single model
    singles = {nm: 1 - (sub[nm] * vw).sum() for nm in ourcols}
    bestnm = max(singles, key=singles.get)
    win_keys = (sub["oracle"] < sub["LEGO"]).mean(); win_vol = vw[sub["oracle"].values < sub["LEGO"].values].sum()
    print(f"\n  {cat}: LEGO acc {accL:.3f} | ORACLE acc {accO:.3f} ({'BEATS' if accO > accL else 'LOSES'}) | best single = {bestnm} {singles[bestnm]:.3f}")
    print(f"    oracle wins {win_keys:.0%} of keys, {win_vol:.0%} of volume vs LEGO")
    # per-model volume-wtd acc
    print("    per-model acc: " + "  ".join(f"{nm} {singles[nm]:.3f}" for nm in sorted(singles, key=singles.get, reverse=True)[:6]))

# top-volume key error structure: level vs timing (FABRIC + HAIR)
print("\n=== top-8 volume keys: actual vs best-DIQ vs LEGO totals (level check) ===")
for cat in ["FABRIC CLEANING", "HAIR CARE"]:
    sub = base[base.Category == cat]
    tot = sub.groupby("cs").agg(actual=("actual", "sum"), lego=("lego", "sum")).nlargest(8, "actual")
    h8 = MODELS["catselect13"]; h8 = lego_eval.attach_forecast(base, h8, pred_col="predicted")
    diq = h8[h8.Category == cat].groupby("cs")["ours"].sum()
    print(f"\n  {cat}:")
    for cs, r in tot.iterrows():
        d = diq.get(cs, np.nan)
        print(f"    {cs}: actual {r.actual:8.0f}  DIQ {d:8.0f} ({d/r.actual-1:+.0%})  LEGO {r.lego:8.0f} ({r.lego/r.actual-1:+.0%})")
