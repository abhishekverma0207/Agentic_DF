"""Test: can a proper per-key RandomForest (LEGO's model class, which we've never built) match LEGO
on FABRIC CLEANING's high-volume keys? Direct multi-horizon RF per key (one RF per horizon h=1..13)
on lag/calendar/rolling features. Compare 13wk accuracy vs LEGO and our deep on the same keys."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestRegressor
from utils import lego_eval

CAT, ORIGIN = "FABRIC CLEANING", 202602
base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
F = pd.read_parquet("artifacts_th_local/_key_features.parquet"); F["cs"] = F["key"].astype(str).str[:-6]; F = F.set_index("cs")
csset = base.loc[base.Category == CAT, "cs"].unique()
sub = F.loc[F.index.isin(csset)]; thr = sub["vol"].quantile(0.70)
hv = set(sub[(sub.vol >= thr) & ((1 - sub.nzfrac) <= 0.35) & (sub.nzfrac * sub.n >= 52)].index)

d = pd.read_parquet("sourcedata/THAILAND/by_category/FABRIC_CLEANING.parquet")
d = d[d["key"].astype(str).str.endswith("_E1016")].copy()
d["cs"] = d["key"].astype(str).str[:-6]; d = d[d["cs"].isin(hv)]
d["year_week"] = pd.to_numeric(d["year_week"], errors="coerce").astype("int64")
d["Actuals"] = pd.to_numeric(d["Actuals"], errors="coerce").fillna(0.0)
d = d.groupby(["cs", "year_week"], as_index=False)["Actuals"].sum().sort_values(["cs", "year_week"])

FUT = list(range(202603, 202616))


def feats(s):
    """build lag/calendar/rolling features on a single key's weekly series (indexed by year_week)."""
    x = pd.DataFrame({"y": s})
    for L in [1, 2, 3, 4, 8, 13, 26, 52]:
        x[f"lag{L}"] = s.shift(L)
    for W in [4, 8, 13]:
        x[f"rm{W}"] = s.shift(1).rolling(W).mean()
    yw = x.index.to_series(); woy = (yw % 100)
    x["woy_sin"] = np.sin(2 * np.pi * woy / 52); x["woy_cos"] = np.cos(2 * np.pi * woy / 52); x["is_q1"] = (woy <= 15).astype(float)
    return x


rows = []
for cs, g in d.groupby("cs"):
    s = g.set_index("year_week")["Actuals"]
    full_idx = sorted(set(s.index) | set(FUT))
    s = s.reindex(full_idx)
    X = feats(s.fillna(0))
    train_mask = (X.index <= ORIGIN)
    for h, fw in enumerate(FUT, 1):
        # direct-MH: X at week w -> y at w+h ; here predict at ORIGIN for target week fw
        yt = s.shift(-h)  # target h ahead
        tr = X[train_mask & yt.reindex(X.index).notna() & (X.index <= ORIGIN - h)]
        ytr = yt.reindex(tr.index)
        Xo = X[X.index == ORIGIN]
        if len(tr) < 30 or Xo.empty:
            continue
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.7, n_jobs=2, random_state=0)
        rf.fit(tr.fillna(0).values, ytr.values)
        pred = float(np.clip(rf.predict(Xo.fillna(0).values)[0], 0, None))
        rows.append((cs + "_E1016", fw, pred))
rf_fc = pd.DataFrame(rows, columns=["key", "year_week", "predicted"])
rf_fc.to_parquet("artifacts_th_local/_perkey_rf_fabric_fc.parquet", index=False)


def acc_bias_on(fc, label):
    cskeys = set(c + "_E1016" for c in hv)
    p = lego_eval.attach_forecast(base, fc[fc.key.isin(cskeys)], pred_col="predicted")
    p = p[(p.Category == CAT) & p.cs.isin(hv)]
    a = p["actual"].values; f = p["ours"].fillna(0).values; t = np.abs(a).sum()
    la = 1 - np.abs(p["lego"] - p["actual"]).sum() / t
    print(f"  {label:16s} acc {1-np.abs(f-a).sum()/t:+.3f}  bias {(f-a).sum()/t:+.3f}   (LEGO acc {la:+.3f} on these {len(hv)} hv keys)")


print(f"=== per-key RF vs LEGO vs deep on FABRIC CLEANING {len(hv)} high-vol keys (13wk) ===")
acc_bias_on(rf_fc, "per-key RF")
acc_bias_on(pd.read_parquet("artifacts_th_local/deep_TFT_Huber8_fc.parquet").assign(key=lambda x: x.key.astype(str)), "deep Huber8")
acc_bias_on(pd.read_parquet("artifacts_th_local/final_forecast_catselect13.parquet").assign(key=lambda x: x.key.astype(str)), "catselect13")
