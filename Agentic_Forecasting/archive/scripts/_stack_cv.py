"""Decisive test: does a STACKING meta-learner GENERALIZE across keys? Build a cell-level dataset
(base-model predictions + key features per cs x week x horizon), 5-fold CV over KEYS, train a
LightGBM to predict the actual from the base preds + features on train-keys, predict held-out keys.
If the out-of-fold stacker beats LEGO on the hard 3, the routing is learnable -> build the leak-free
version. (Uses test-origin forecasts, so this measures cross-KEY generalization, not yet cross-time.)"""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
base["year_week"] = base["Shipment_Week"].str.replace("-", "").astype(int)
base["key"] = base["cs"] + "_E1016"
HARD = ["FABRIC CLEANING", "HAIR CARE", "FOODS"]

MODELS = {}
for f in sorted(glob.glob("artifacts_th_local/deep_TFT_*_fc.parquet")):
    nm = f.split("deep_TFT_")[1].split("_fc")[0]
    if nm != "allkeys":
        d = pd.read_parquet(f); MODELS[nm] = d
for nm, path in [("perkey", "artifacts_th_local/final_forecast_router.parquet"),
                 ("seasonal", "artifacts_th_local/_seasonal_perkey_fc.parquet"),
                 ("SPLY", "artifacts_th_local/_sply_naive_fc.parquet")]:
    MODELS[nm] = pd.read_parquet(path)
PREDS = list(MODELS)

# wide cell-level frame
df = base[["key", "cs", "Category", "year_week", "t", "actual", "lego"]].copy()
for nm, m in MODELS.items():
    m = m.copy(); m["key"] = m["key"].astype(str); m["year_week"] = m["year_week"].astype(int)
    df = df.merge(m[["key", "year_week", "predicted"]].rename(columns={"predicted": nm}), on=["key", "year_week"], how="left")
F = pd.read_parquet("artifacts_th_local/_key_features.parquet"); F["key"] = F["key"].astype(str); F = F.set_index("key")
for c in ["vol", "mean", "nzfrac", "ADI", "CV2", "spike", "recent", "n"]:
    df[f"f_{c}"] = df["key"].map(F[c])
df["f_logvol"] = np.log1p(df["f_vol"]); df["f_logmean"] = np.log1p(df["f_mean"])
df = df[df.Category.isin(HARD)].copy()
for nm in PREDS:
    df[nm] = df[nm].fillna(0.0)
FEATS = PREDS + ["t", "f_logvol", "f_logmean", "f_nzfrac", "f_ADI", "f_CV2", "f_spike", "f_recent", "f_n"]

# 5-fold CV over keys -> out-of-fold stacked prediction
df["stack"] = np.nan
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(df, groups=df["cs"]):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1)
    m.fit(df.iloc[tr][FEATS], df.iloc[tr]["actual"])
    df.iloc[te, df.columns.get_loc("stack")] = np.clip(m.predict(df.iloc[te][FEATS]), 0, None)


def acc_bias(g, col):
    a = g["actual"].values; f = g[col].fillna(0).values; t = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(t, 1e-9), (f - a).sum() / max(t, 1e-9))


print("=== out-of-fold STACK (LightGBM, CV over keys) vs LEGO + best single, 13wk ===")
for cat in HARD:
    g = df[df.Category == cat]
    sa, sb = acc_bias(g, "stack"); la, lb = acc_bias(g, "lego")
    singles = {nm: acc_bias(g, nm)[0] for nm in PREDS}; bestnm = max(singles, key=singles.get)
    v = "WIN both" if (sa > la and abs(sb) < abs(lb)) else ("acc" if sa > la else ("bias" if abs(sb) < abs(lb) else "-"))
    print(f"  {cat:18s} STACK acc {sa:+.3f} bias {sb:+.3f} | LEGO acc {la:+.3f} bias {lb:+.3f} | best-single {bestnm} {singles[bestnm]:.3f}  => {v}")
# feature importance from a full-data fit (which models/features the stacker leans on)
m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, verbosity=-1).fit(df[FEATS], df["actual"])
imp = sorted(zip(FEATS, m.feature_importances_), key=lambda z: -z[1])[:10]
print("  stacker leans on:", {k: int(v) for k, v in imp})
