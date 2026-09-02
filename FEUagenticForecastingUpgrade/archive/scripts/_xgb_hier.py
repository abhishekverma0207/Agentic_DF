"""Hierarchical XGBoost key-level model — the 'use all the data' lever. Global direct-multi-horizon
XGBoost over E1016 keys with HIERARCHICAL features the deep/per-key models lack:
  - own E1016 history (lags, rolling, SPLY)
  - CROSS-ACCOUNT: the same CS_BARCODE summed across all ~50 accounts (lags, SPLY, trend) -> the 68%
    of each product's demand we were discarding (level/trend/seasonality signal)
  - CATEGORY aggregates (total, SPLY, trend)
  - known-future calendar + forecastability statics
Leak-free: every feature is as-of origin week w (<= w); targets used in training are <= test origin
202602. Trains once, forecasts 202603..202615, evals per category 13wk vs LEGO.
Usage: _xgb_hier.py [cats=all|hard] [start_origin=202301]"""
import os, sys, time
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import xgboost as xgb
from utils import lego_eval

t0 = time.time()
WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"
START = int(sys.argv[2]) if len(sys.argv) > 2 else 202315   # earliest ORIGIN week used for training rows
ORIGIN = int(os.environ.get("XGB_ORIGIN", "202602"))   # last-known week; forecast ORIGIN+1..+13 (rolling-origin lock)
DECAY = float(os.environ.get("XGB_DECAY", "1.0"))           # recency decay per year (1.0 = off)
ALLCATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
           "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]
HARD = ["FABRIC_CLEANING", "HAIR_CARE", "FOODS"]
CATS = HARD if WHICH == "hard" else ALLCATS

# ---- full continuous weekly grid (ISO weeks) ----
grid = pd.date_range("2022-01-03", "2026-04-13", freq="W-MON")
gw = grid.isocalendar(); weeks = (gw["year"].astype(int) * 100 + gw["week"].astype(int)).tolist()
weeks = sorted(set(weeks)); pos = {w: i for i, w in enumerate(weeks)}; NW = len(weeks)
Opos = pos[ORIGIN]

# ---- load all-account data; build own(E1016), xacct(all-account total), cat totals ----
frames = []
for c in CATS:
    cols = ["key", "year_week", "Actuals", "IsPromo", "Discount_Percent", "promo_seq_num",
            "PromoTionType_On_Invoice", "PromoTionType_Off_Invoice"]
    try:
        d = pd.read_parquet(f"sourcedata/THAILAND/by_category/{c}.parquet", columns=cols)
    except Exception:
        d = pd.read_parquet(f"sourcedata/THAILAND/by_category/{c}.parquet")
        d = d[[x for x in cols if x in d.columns]]
    d["Category"] = c; frames.append(d)
raw = pd.concat(frames, ignore_index=True)
for pc in ["IsPromo", "Discount_Percent", "promo_seq_num", "PromoTionType_On_Invoice", "PromoTionType_Off_Invoice"]:
    raw[pc] = pd.to_numeric(raw.get(pc), errors="coerce").fillna(0.0) if pc in raw.columns else 0.0
raw["year_week"] = pd.to_numeric(raw["year_week"], errors="coerce").astype("Int64")
raw = raw[raw["year_week"].isin(set(weeks))].copy()
raw["Actuals"] = pd.to_numeric(raw["Actuals"], errors="coerce").fillna(0.0).clip(lower=0)
raw["cs"] = raw["key"].astype(str).str.rsplit("_", n=1).str[0]
raw["acct"] = raw["key"].astype(str).str.rsplit("_", n=1).str[1]
raw["wp"] = raw["year_week"].map(pos).astype("int64")

keys = sorted(raw.loc[raw.acct == "E1016", "cs"].unique())
kpos = {k: i for i, k in enumerate(keys)}; NK = len(keys)
cs2cat = raw.drop_duplicates("cs").set_index("cs")["Category"]
catids = sorted(CATS); catid = {c: i for i, c in enumerate(catids)}
print(f"[xgb] {NK} E1016 keys, {len(CATS)} cats, {NW} weeks ({time.time()-t0:.0f}s)", flush=True)


def to_arr(df, rowmap, nrows):
    a = np.zeros((nrows, NW), dtype=np.float32)
    df = df.dropna(subset=["__r"])
    r = df["__r"].astype(int).values; cc = df["wp"].astype(int).values; vv = df["Actuals"].values.astype(np.float32)
    np.add.at(a, (r, cc), vv); return a


own_df = raw[raw.acct == "E1016"].copy(); own_df["__r"] = own_df["cs"].map(kpos)
own = to_arr(own_df.dropna(subset=["__r"]), kpos, NK)
xa_df = raw.groupby(["cs", "wp"], as_index=False)["Actuals"].sum(); xa_df["__r"] = xa_df["cs"].map(kpos)
xacct = to_arr(xa_df.dropna(subset=["__r"]), kpos, NK)
cat_df = raw[raw.acct == "E1016"].copy(); cat_df["__c"] = cat_df["Category"].map(catid); cat_df["__r"] = cat_df["__c"]
catarr = np.zeros((len(catids), NW), dtype=np.float32)
np.add.at(catarr, (cat_df["__r"].values, cat_df["wp"].values), cat_df["Actuals"].values)
key_cat = np.array([catid[cs2cat[k]] for k in keys])

# promo arrays (E1016, known-future: promo plans are set ahead) -> max IsPromo, mean Discount per key-week
pr_g = raw[raw.acct == "E1016"].groupby(["cs", "wp"]).agg(
    p=("IsPromo", "max"), dsc=("Discount_Percent", "mean"), seq=("promo_seq_num", "max"),
    pon=("PromoTionType_On_Invoice", "max"), poff=("PromoTionType_Off_Invoice", "max")).reset_index()
pr_g["__r"] = pr_g["cs"].map(kpos)
pr_arr = np.zeros((NK, NW), dtype=np.float32); disc_arr = np.zeros((NK, NW), dtype=np.float32)
pseq_arr = np.zeros((NK, NW), dtype=np.float32); pon_arr = np.zeros((NK, NW), dtype=np.float32); poff_arr = np.zeros((NK, NW), dtype=np.float32)
_pg = pr_g.dropna(subset=["__r"]); _r = _pg["__r"].astype(int).values; _c = _pg["wp"].astype(int).values
pr_arr[_r, _c] = _pg["p"].values.astype(np.float32); disc_arr[_r, _c] = _pg["dsc"].values.astype(np.float32)
pseq_arr[_r, _c] = _pg["seq"].values.astype(np.float32); pon_arr[_r, _c] = _pg["pon"].values.astype(np.float32); poff_arr[_r, _c] = _pg["poff"].values.astype(np.float32)

# DISTRIBUTION BREADTH: # distinct accounts active per CS_BARCODE per week (the portfolio-growth signal
# a per-key model can't see — expanding distribution leads demand growth). Built from the all-account panel.
an = raw[raw.Actuals > 0].groupby(["cs", "wp"])["acct"].nunique().reset_index(); an["__r"] = an["cs"].map(kpos)
acctn_arr = np.zeros((NK, NW), dtype=np.float32); _an = an.dropna(subset=["__r"])
acctn_arr[_an["__r"].astype(int).values, _an["wp"].astype(int).values] = _an["acct"].values.astype(np.float32)

# forecastability statics
F = pd.read_parquet("artifacts_th_local/_key_features.parquet"); F["cs"] = F["key"].astype(str).str[:-6]; F = F.set_index("cs")
ST = {}
for c in ["vol", "mean", "nzfrac", "ADI", "CV2", "spike", "recent"]:
    ST[c] = np.array([F[c].get(k, np.nan) for k in keys], dtype=np.float32)
ST["logvol"] = np.log1p(ST["vol"]); ST["logmean"] = np.log1p(ST["mean"])
# volume tertile within category — for SEGMENT-smear (high-vol cells, which grew, get the right uplift)
key_vseg = np.zeros(NK, dtype=int); _v = np.nan_to_num(ST["vol"])
for ci in range(len(catids)):
    m = key_cat == ci
    if m.sum() >= 6:
        q = np.quantile(_v[m], [1 / 3, 2 / 3]); key_vseg[m] = np.digitize(_v[m], q)


def roll_mean(a, w, end):  # mean of a[:, end-w+1 .. end] along time
    lo = max(0, end - w + 1); return a[:, lo:end + 1].mean(axis=1)


def roll_std(a, w, end):
    lo = max(0, end - w + 1); return a[:, lo:end + 1].std(axis=1)


def slope(a, w, end):  # OLS slope over last w points = the demand 'curve' gradient
    lo = max(0, end - w + 1); seg = a[:, lo:end + 1]; n = seg.shape[1]
    if n < 2:
        return np.zeros(a.shape[0], dtype=np.float32)
    x = np.arange(n, dtype=np.float32); xm = x.mean(); xx = ((x - xm) ** 2).sum()
    return ((seg - seg.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / (xx + 1e-9)


def col_at(a, idx, nk):  # safe column slice (zeros if out of range)
    return a[:, idx] if idx >= 0 else np.zeros(nk, np.float32)


def build_rows(origins):
    """for each origin wp in origins and horizon h, build leak-free features (as-of wp) + target own[wp+h]."""
    X, Y = [], []
    woy = np.array([w % 100 for w in weeks])
    for wp in origins:
        for h in range(1, 14):
            tp = wp + h
            if tp >= NW:
                continue
            train_ok = tp <= Opos  # target observed by the test origin (leak-free for training)
            if origins is TRAIN_ORIGINS and not train_ok:
                continue
            sp = tp - 52  # SPLY position (same week last year as the TARGET)
            f = {
                "h": h,
                "own0": own[:, wp], "own1": own[:, max(0, wp - 1)], "own3": own[:, max(0, wp - 3)],
                "own7": own[:, max(0, wp - 7)], "own12": own[:, max(0, wp - 12)], "own52": own[:, max(0, wp - 51)],
                "own_rm4": roll_mean(own, 4, wp), "own_rm8": roll_mean(own, 8, wp), "own_rm13": roll_mean(own, 13, wp),
                "own_sply": own[:, sp] if sp >= 0 else np.zeros(NK, np.float32),
                "xa0": xacct[:, wp], "xa3": xacct[:, max(0, wp - 3)], "xa52": xacct[:, max(0, wp - 51)],
                "xa_rm8": roll_mean(xacct, 8, wp), "xa_sply": xacct[:, sp] if sp >= 0 else np.zeros(NK, np.float32),
                "xa_trend": np.log1p(roll_mean(xacct, 8, wp)) - np.log1p(roll_mean(xacct, 8, max(0, wp - 52))),
                "cat0": catarr[key_cat, wp], "cat_sply": catarr[key_cat, sp] if sp >= 0 else np.zeros(NK, np.float32),
                "cat_trend": np.log1p(roll_mean(catarr, 8, wp))[key_cat] - np.log1p(roll_mean(catarr, 8, max(0, wp - 52)))[key_cat],
                "share": roll_mean(own, 8, wp) / (roll_mean(xacct, 8, wp) + 1.0),
                "woy_sin": np.full(NK, np.sin(2 * np.pi * woy[tp] / 52)), "woy_cos": np.full(NK, np.cos(2 * np.pi * woy[tp] / 52)),
                "is_q1": np.full(NK, 1.0 if woy[tp] <= 15 else 0.0),
                "pr_tgt": pr_arr[:, tp], "disc_tgt": disc_arr[:, tp], "pr_now": pr_arr[:, wp],
                "pr_sply": col_at(pr_arr, sp, NK),
                # --- FORWARD-PROMO CAMPAIGN intelligence (planned promo over the horizon = the growth signal) ---
                "promo_int_tgt": pr_arr[:, tp] * disc_arr[:, tp], "pseq_tgt": pseq_arr[:, tp],
                "pon_tgt": pon_arr[:, tp], "poff_tgt": poff_arr[:, tp],
                "promo_path_n": pr_arr[:, wp + 1:tp + 1].sum(axis=1),                                # # planned promo wks origin->target
                "promo_path_disc": (pr_arr[:, wp + 1:tp + 1] * disc_arr[:, wp + 1:tp + 1]).sum(axis=1),  # cumulative planned intensity
                "promo_win3": pr_arr[:, max(0, tp - 1):min(NW, tp + 2)].sum(axis=1),                 # promo around target (+/-1 wk)
                "disc_tgt_yoy": disc_arr[:, tp] - col_at(disc_arr, sp, NK),                          # planned vs last-yr discount
                "promo_vs_sply": pr_arr[:, tp] - col_at(pr_arr, sp, NK),                             # more/less promo than last year
                # --- growth / momentum curves (the FABRIC growth lever) ---
                "own_yoy": roll_mean(own, 8, wp) / (roll_mean(own, 8, max(0, wp - 52)) + 1.0),
                "xa_yoy": roll_mean(xacct, 8, wp) / (roll_mean(xacct, 8, max(0, wp - 52)) + 1.0),
                "cat_yoy": (roll_mean(catarr, 8, wp) / (roll_mean(catarr, 8, max(0, wp - 52)) + 1.0))[key_cat],
                "own_slope13": slope(own, 13, wp), "own_slope26": slope(own, 26, wp), "xa_slope13": slope(xacct, 13, wp),
                "own_mom": roll_mean(own, 4, wp) / (roll_mean(own, 13, wp) + 1.0),
                "own_accel": (roll_mean(own, 4, wp) - roll_mean(own, 8, wp)) - (roll_mean(own, 8, wp) - roll_mean(own, 13, wp)),
                "recent_hist": roll_mean(own, 8, wp) / (roll_mean(own, 52, wp) + 1.0),
                # --- seasonality curves ---
                "woy_sin2": np.full(NK, np.sin(4 * np.pi * woy[tp] / 52)), "woy_cos2": np.full(NK, np.cos(4 * np.pi * woy[tp] / 52)),
                "woy_sin3": np.full(NK, np.sin(6 * np.pi * woy[tp] / 52)), "woy_cos3": np.full(NK, np.cos(6 * np.pi * woy[tp] / 52)),
                "seas_idx": own[:, wp] / (roll_mean(own, 52, wp) + 1.0),
                "own_sply2": col_at(own, tp - 104, NK),
                "sply_yoy": (own[:, sp] / (own[:, tp - 104] + 1.0)) if (sp >= 0 and tp - 104 >= 0) else np.ones(NK, np.float32),
                "cat_sply_tr": (col_at(catarr, sp, len(catids)) / (col_at(catarr, tp - 104, len(catids)) + 1.0))[key_cat] if (sp >= 0 and tp - 104 >= 0) else np.ones(NK, np.float32),
                "woy": np.full(NK, float(woy[tp])),
                # --- level / volatility ---
                "own_rm26": roll_mean(own, 26, wp), "own_rm52": roll_mean(own, 52, wp),
                "own_std8": roll_std(own, 8, wp), "own_std13": roll_std(own, 13, wp),
                "own_max13": own[:, max(0, wp - 12):wp + 1].max(axis=1), "own_min13": own[:, max(0, wp - 12):wp + 1].min(axis=1),
                # --- promo curves + hierarchy dynamics ---
                "pr_rate8": roll_mean(pr_arr, 8, wp), "disc_now": disc_arr[:, wp], "disc_rm8": roll_mean(disc_arr, 8, wp),
                "share_tr": (roll_mean(own, 8, wp) / (roll_mean(xacct, 8, wp) + 1.0)) / ((roll_mean(own, 8, max(0, wp - 13)) / (roll_mean(xacct, 8, max(0, wp - 13)) + 1.0)) + 1e-6),
                # --- distribution breadth + growth acceleration (momentum = the proven growth predictors) ---
                "acctn": roll_mean(acctn_arr, 8, wp), "acctn_now": acctn_arr[:, wp],
                "acctn_yoy": roll_mean(acctn_arr, 8, wp) / (roll_mean(acctn_arr, 8, max(0, wp - 52)) + 0.1),
                "acctn_tr": roll_mean(acctn_arr, 4, wp) / (roll_mean(acctn_arr, 13, wp) + 0.1),
                "yoy_accel": (roll_mean(own, 8, wp) / (roll_mean(own, 8, max(0, wp - 52)) + 1.0)) - (roll_mean(own, 8, max(0, wp - 13)) / (roll_mean(own, 8, max(0, wp - 65)) + 1.0)),
                "xa_yoy_lead": roll_mean(xacct, 4, wp) / (roll_mean(xacct, 4, max(0, wp - 52)) + 1.0),
                "own_yoy4": roll_mean(own, 4, wp) / (roll_mean(own, 4, max(0, wp - 52)) + 1.0),
                "catid": key_cat.astype(np.float32),
                **{f"st_{c}": ST[c] for c in ["logvol", "logmean", "nzfrac", "ADI", "CV2", "spike", "recent"]},
            }
            fr = pd.DataFrame(f)
            fr["__keyidx"] = np.arange(NK, dtype=np.int32)   # row -> key (for per-key models)
            # ratio-model baseline b: SPLY (same-season-last-year) with fallback to recent level for low-SPLY keys
            _b = np.where(fr["own_sply"].values > 0.2 * fr["own_rm8"].values + 1.0, fr["own_sply"].values, fr["own_rm8"].values)
            fr["__b"] = np.maximum(_b, 1.0).astype(np.float32)
            fr["__seg"] = key_cat * 3 + key_vseg   # (category, volume-tertile) smear segment
            if origins is TRAIN_ORIGINS:
                fr["__y"] = own[:, tp]
                fr = fr[(fr["own_rm13"] > 0) | (fr["own0"] > 0)]  # drop fully-dead-history rows
                fr["__wt"] = float(DECAY ** ((Opos - wp) / 52.0))   # recency weight (lean toward the current regime)
            else:
                fr["__k"] = keys; fr["__tw"] = weeks[tp]
            X.append(fr)
    return pd.concat(X, ignore_index=True)


TRAIN_ORIGINS = [pos[w] for w in weeks if START <= w <= ORIGIN]
print(f"[xgb] building training rows from {len(TRAIN_ORIGINS)} origins ({time.time()-t0:.0f}s)", flush=True)
tr = build_rows(TRAIN_ORIGINS)
FEATS = [c for c in tr.columns if c not in ("__y", "__k", "__tw", "__wt", "__b", "__seg", "__keyidx")]
print(f"[xgb] train {len(tr):,} rows x {len(FEATS)} feats; training XGBoost ({time.time()-t0:.0f}s)", flush=True)
OBJ = os.environ.get("XGB_OBJ", "huber")
SMEAR = os.environ.get("XGB_SMEAR", "1") == "1"
uselog = OBJ in ("log", "logsmear")
useratio = OBJ == "ratio"   # predict log((y+1)/(b+1)); baseline b carries the level, trees learn the growth multiplier
ALPHA = float(os.environ.get("XGB_QALPHA", "0"))  # >0 -> quantile objective (counter median under-bias)
# hyperparameters env-tunable (for a dedicated higher-accuracy per-category base)
_E = os.environ
mkw = dict(n_estimators=int(_E.get("XGB_NEST", "700")), learning_rate=float(_E.get("XGB_LR", "0.04")),
           max_depth=int(_E.get("XGB_DEPTH", "8")), subsample=float(_E.get("XGB_SUB", "0.8")),
           colsample_bytree=float(_E.get("XGB_COL", "0.7")), min_child_weight=float(_E.get("XGB_MCW", "20")),
           reg_lambda=float(_E.get("XGB_LAMBDA", "2.0")), tree_method="hist", n_jobs=0)
LEARNER = _E.get("XGB_LEARNER", "xgb")   # 'xgb' | 'lgb' (LightGBM = a diverse GBM family for ensembling)
if ALPHA > 0:
    mkw.update(objective="reg:quantileerror", quantile_alpha=ALPHA)
elif OBJ == "tweedie":
    mkw.update(objective="reg:tweedie", tweedie_variance_power=1.3)
elif OBJ == "rmse":
    mkw.update(objective="reg:squarederror")   # plain L2 -> fits the volume-weighted mean (accuracy-first)
elif OBJ in ("huber", "ratio"):
    mkw.update(objective="reg:pseudohubererror")


def fwd(y, b):   # target transform
    return np.log((y + 1.0) / (b + 1.0)) if useratio else (np.log1p(y) if uselog else y)


def inv(p, b):   # inverse transform back to units
    return ((b + 1.0) * np.exp(np.clip(p, -5, 5)) - 1.0) if useratio else (np.expm1(p) if uselog else p)


def make_model(mkw, learner):
    """Global learner: XGBoost, or LightGBM (a diverse GBM family — leaf-wise growth, different
    regularization — for ensembling to push accuracy past a single model's ceiling)."""
    if learner == "lgb":
        import lightgbm as lgb
        obj = mkw.get("objective", "")
        lmap = {"reg:squarederror": "regression", "reg:pseudohubererror": "huber",
                "reg:tweedie": "tweedie", "reg:quantileerror": "quantile"}
        lkw = dict(n_estimators=mkw["n_estimators"], learning_rate=mkw["learning_rate"],
                   max_depth=mkw["max_depth"], num_leaves=min(2 ** mkw["max_depth"] - 1, 255),
                   subsample=mkw["subsample"], subsample_freq=1, colsample_bytree=mkw["colsample_bytree"],
                   min_child_weight=mkw["min_child_weight"], reg_lambda=mkw["reg_lambda"],
                   objective=lmap.get(obj, "regression"), n_jobs=0, verbosity=-1)
        if obj == "reg:tweedie":
            lkw["tweedie_variance_power"] = 1.3
        if obj == "reg:quantileerror":
            lkw["alpha"] = mkw.get("quantile_alpha", 0.5)
        return lgb.LGBMRegressor(**lkw)
    return xgb.XGBRegressor(**mkw)


model = make_model(mkw, LEARNER)
model.fit(tr[FEATS], fwd(tr["__y"].values, tr["__b"].values),
          sample_weight=tr["__wt"].values if DECAY != 1.0 else None)
# per-category multiplicative level correction (Duan-style smearing), learned on TRAIN only -> leak-free.
SEGSMEAR = os.environ.get("XGB_SEGSMEAR", "0") == "1"  # smear per (category x volume-tertile), not just category
smkey = "__seg" if SEGSMEAR else "catid"
smear = {}
if SMEAR:
    trp = np.clip(inv(model.predict(tr[FEATS]), tr["__b"].values), 0, None)
    tmp = pd.DataFrame({"c": tr[smkey].values.astype(int), "y": tr["__y"].values, "p": trp})
    SMAX = float(os.environ.get("XGB_SMAX", "1.6"))
    smear = tmp.groupby("c").apply(lambda g: float(np.clip(g["y"].sum() / max(g["p"].sum(), 1e-9), 0.7, SMAX))).to_dict()
PERKEY = os.environ.get("XGB_PERKEY", "0") == "1"

te = build_rows([Opos])
pred = np.clip(inv(model.predict(te[FEATS]), te["__b"].values), 0, None)
if SMEAR:
    pred = pred * np.array([smear.get(int(c), 1.0) for c in te[smkey].values])

# --- PER-KEY models (LEGO's approach): a regularized XGBoost per high-volume key on its OWN rows
# (full rich features), per-key level smear, blended with the global model (hedge against overfit) ---
if PERKEY:
    PKW = float(os.environ.get("XGB_PKW", "0.6")); PKMIN = int(os.environ.get("XGB_PKMIN", "120"))
    pkobj = dict(objective="reg:quantileerror", quantile_alpha=ALPHA) if ALPHA > 0 else dict(objective="reg:pseudohubererror")
    pkcfg = dict(n_estimators=250, max_depth=4, learning_rate=0.05, min_child_weight=12,
                 reg_lambda=5.0, subsample=0.7, colsample_bytree=0.5, tree_method="hist", n_jobs=1, **pkobj)
    te_ki = te["__keyidx"].values
    hv = [int(k) for k in np.where(key_vseg == 2)[0]]   # top volume-tertile keys within each category
    grp = {int(k): g for k, g in tr.groupby("__keyidx") if int(k) in set(hv)}
    npk = 0
    for k in hv:
        g = grp.get(k); mask = te_ki == k
        if g is None or len(g) < PKMIN or mask.sum() == 0:
            continue
        m = xgb.XGBRegressor(**pkcfg)
        m.fit(g[FEATS], fwd(g["__y"].values, g["__b"].values))
        tem = te[mask]; gp = pred[mask]   # the sane global+smear prediction for these cells
        pk = np.clip(inv(m.predict(tem[FEATS]), tem["__b"].values), 0, None)
        trp = np.clip(inv(m.predict(g[FEATS]), g["__b"].values), 0, None)
        sf = float(np.clip(g["__y"].sum() / max(trp.sum(), 1e-9), 0.6, 2.0))   # per-key level smear
        pk = np.minimum(pk * sf, 3.0 * gp + 5.0)   # bound per-key extrapolation to the reliable global
        pred[mask] = PKW * pk + (1 - PKW) * gp
        npk += 1
    print(f"[xgb] trained {npk} per-key models (blend w={PKW}, min_rows={PKMIN})", flush=True)
del tr
out = pd.DataFrame({"key": te["__k"].astype(str) + "_E1016", "year_week": te["__tw"].astype(int), "predicted": pred})
out.to_parquet(os.environ.get("XGB_OUT", "artifacts_th_local/_xgb_hier_fc.parquet"), index=False)
print(f"[xgb] forecast saved ({len(out)} rows, {time.time()-t0:.0f}s)", flush=True)

# ---- eval (only meaningful at the test origin; LEGO benchmark exists for 202603-15 only) ----
if ORIGIN != 202602:
    print(f"[xgb] ORIGIN={ORIGIN} (rolling-origin lock) -> skipping LEGO eval; forecast saved.", flush=True)
    sys.exit(0)
base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
EVN = {c: c.replace("_AND_", " & ").replace("_", " ") for c in CATS}
EVN["DEODORANTS_AND_FRAGRANCES"] = "DEODORANTS & FRAGRANCES"


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; t = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(t, 1e-9), (f - a).sum() / max(t, 1e-9))


p = lego_eval.attach_forecast(base, out, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
TOP5 = {"FABRIC CLEANING", "HOME & HYGIENE", "HAIR CARE", "FOODS", "FABRIC ENHANCERS"}
print("=== HIERARCHICAL XGBoost (13wk) DIQ acc/LEGO  bias/LEGO ===")
for c in sorted(p.Category.unique()):
    pc = p[p.Category == c]; a, b = ab(pc); la, lb = ab(pc.assign(ours=pc.lego))
    v = "WIN both" if (a > la and abs(b) < abs(lb)) else ("acc" if a > la else ("bias" if abs(b) < abs(lb) else "-"))
    print(f"  {c:24s} acc {a:+.3f}/{la:+.3f}  bias {b:+.3f}/{lb:+.3f}  {v}{'  *top5' if c in TOP5 else ''}")
