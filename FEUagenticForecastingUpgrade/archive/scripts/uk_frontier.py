#!/usr/bin/env python3
"""
Frontier evidence: is +2pp over LEGO achievable, or are we at the data's lag-4
predictability ceiling? Four independent angles, all at GTIN x week (the eval grain):

  (1) Naive baselines  — how much accuracy is "free" (recent average / SPLY / last value)?
  (2) LEGO by horizon  — does accuracy decay with lag (information limit)?
  (3) Oracle ceiling   — irreducible week-to-week noise (best any level-forecast can do).
  (4) Model families   — LightGBM vs XGBoost vs RandomForest vs Ridge on identical features.

Run: .venv/bin/python scripts/uk_frontier.py
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import uk_eval as E, uk_forecast as F

SN = ["202609", "202611", "202613", "202615"]
def acc(f, a):
    a = np.asarray(a, float); f = np.nan_to_num(np.asarray(f, float)); sa = a.sum()
    return np.nan if sa <= 0 else 1 - np.abs(f - a).sum() / sa

# gtin x week actuals (all weeks, for baselines + oracle)
A = E._read_actuals(); A["cs_gtin"] = A["key"].astype(str).str.rsplit("_", n=1).str[0]
ga = A.groupby(["cs_gtin", "year_week"], as_index=False)["actual"].sum()
gd = {(g, w): v for g, w, v in zip(ga.cs_gtin, ga.year_week, ga.actual)}

fr = E.load_scoring_frame(SN, lags=(4,))
glob = pd.read_parquet("artifacts_uk_local/experiments_acc/forecasts_global.parquet")
fr = E.attach_candidate(fr, glob, value_col="forecast", name="ours"); fr["ours"] = fr["ours"].fillna(0)
g = fr.groupby(["category_name", "cs_gtin", "year_week", "snapshot"], as_index=False).agg(
    actual=("actual", "sum"), rf=("prediction_rf", "sum"), ours=("ours", "sum"))

# (1) naive baselines on the lag-4 target set
def naive(row, kind):
    gt, yw, snap = row.cs_gtin, row.year_week, row.snapshot
    te = F.add_weeks(snap, -1)
    if kind == "rmean13":
        vs = [gd.get((gt, F.add_weeks(te, -i)), 0.0) for i in range(13)]
        return float(np.mean(vs))
    if kind == "sply":   return gd.get((gt, F.add_weeks(yw, -52)), 0.0)
    if kind == "lastval":return gd.get((gt, te), 0.0)
for k in ["rmean13", "sply", "lastval"]:
    g[k] = [naive(r, k) for r in g.itertuples()]
print("=== (1) NAIVE BASELINES vs ours vs LEGO  (GTIN lag-4 accuracy, pooled) ===")
for c in ["lastval", "sply", "rmean13", "ours", "rf"]:
    print(f"   {c:9s} {acc(g[c], g.actual):.4f}")

# (2) LEGO accuracy by horizon (lag 1..13)
print("\n=== (2) LEGO prediction_rf accuracy BY HORIZON (information decay) ===")
allfr = E.load_scoring_frame(SN, lags=tuple(range(1, 14)))
gh = allfr.groupby(["cs_gtin", "year_week", "horizon"], as_index=False).agg(
    actual=("actual", "sum"), rf=("prediction_rf", "sum"))
for h in range(1, 14):
    sub = gh[gh.horizon == h]
    print(f"   lag {h:2d}: LEGO acc {acc(sub.rf, sub.actual):.4f}")

# (3) oracle ceiling: centered smooth of actual (uses future => oracle) vs raw, on lag-4 weeks
gaa = ga.sort_values(["cs_gtin", "year_week"]).copy()
gaa["smooth5"] = gaa.groupby("cs_gtin")["actual"].transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
sm = {(r.cs_gtin, r.year_week): r.smooth5 for r in gaa.itertuples()}
g["oracle"] = [sm.get((r.cs_gtin, r.year_week), 0.0) for r in g.itertuples()]
print("\n=== (3) ORACLE ceiling (perfect smoothed-level forecast; irreducible noise) ===")
print(f"   oracle smoothed-actual acc {acc(g.oracle, g.actual):.4f}   <- upper bound for any level forecast")

# (4) model families on identical lag-4 features (one origin, sampled)
print("\n=== (4) MODEL FAMILIES on identical lag-4 features (origin 202613, GTIN lag-4 acc) ===")
import lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
cfg = F.load_config(overrides={"bias_calibration": False})
o = "202613"; TGT = cfg["target"]; te = F.add_weeks(o, -1)
df = F.read_input_panel(o).copy(); df[TGT] = pd.to_numeric(df[TGT], errors="coerce").fillna(0)
fut = F._future_cols(df, cfg); back = F.add_backward_features(df, cfg); catc = F._encode_categoricals(df, cfg)
feat = back + catc + fut
weeks = sorted(df.year_week.unique())
left = df[["key", "year_week"] + back + catc].copy(); right = df[["key", "year_week"] + fut + [TGT]].rename(columns={"year_week": "_tw", TGT: "_y"})
left["_tw"] = left.year_week.map({w: F.add_weeks(w, 4) for w in weeks})
P = left.merge(right, on=["key", "_tw"], how="inner")
tr = P[P["_tw"] <= te]; pr = P[P.year_week == te].copy()
trs = tr.sample(n=min(400000, len(tr)), random_state=0)
Xtr, ytr = trs[feat].fillna(0), trs["_y"].clip(lower=0)
Xpr = pr[feat].fillna(0)
def score_fam(pred):
    pr2 = pr[["key", "_tw"]].copy(); pr2["f"] = np.clip(pred, 0, None); pr2["year_week"] = pr2["_tw"].astype(str)
    pr2["cs_gtin"] = pr2.key.str.rsplit("_", n=1).str[0]
    j = pr2.groupby(["cs_gtin", "year_week"], as_index=False).f.sum().merge(
        ga, on=["cs_gtin", "year_week"], how="left").fillna(0)
    return acc(j.f, j.actual)
lgbm = lgb.train(F._lgb_params(cfg), lgb.Dataset(Xtr, label=ytr, categorical_feature=catc, free_raw_data=False), num_boost_round=400)
print(f"   LightGBM     {score_fam(lgbm.predict(Xpr)):.4f}")
xg = xgb.XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, objective="reg:squarederror")
xg.fit(Xtr, ytr); print(f"   XGBoost      {score_fam(xg.predict(Xpr)):.4f}")
rf = RandomForestRegressor(n_estimators=120, max_depth=14, n_jobs=-1, random_state=0).fit(Xtr.sample(n=min(150000, len(Xtr)), random_state=0), ytr.loc[Xtr.sample(n=min(150000, len(Xtr)), random_state=0).index])
print(f"   RandomForest {score_fam(rf.predict(Xpr)):.4f}")
rg = Ridge(alpha=10.0).fit(Xtr, ytr); print(f"   Ridge        {score_fam(rg.predict(Xpr)):.4f}")
# LEGO on the same origin/lag for reference
ref = gh[(gh.horizon == 4)].merge(g[g.snapshot == o][["cs_gtin", "year_week"]], on=["cs_gtin", "year_week"])
print(f"   LEGO (202613, ref) {acc(ref.rf, ref.actual):.4f}")
