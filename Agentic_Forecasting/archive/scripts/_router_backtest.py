"""FFORMS-style smart router: decide PER KEY whether to use its own per-key model or the
pooled deep model. Backtest both on a held-out window (origin 202541 -> predict 202542-202602,
known actuals) for a stratified set of candidate keys, label each (own beats pooled?), train a
RandomForest on forecastability features, and assign EVERY key. Saves _router_assignment.parquet.
Set SAMPLE=1 for a tiny smoke run."""
import os, sys, glob, time
for k, v in {"MLFLOW_AUTOLOGGING_DISABLE": "1", "OTEL_SDK_DISABLED": "true", "CREWAI_DISABLE_TELEMETRY": "true",
             "TOKENIZERS_PARALLELISM": "false", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import gc
import numpy as np, pandas as pd, logging
import utils.hybrid_forecast as hf  # NOTE: do NOT import utils.deep_forecast here -> keeps torch
from sklearn.ensemble import RandomForestClassifier  # out of this process (torch+LightGBM OMP clash)
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.ERROR, stream=sys.stdout, force=True)
SMOKE = os.environ.get("SMOKE") == "1"
CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]
BT_ORIGIN = 202541           # fit <=202541, predict the validation window 202542..202602
ID = ["key", "year_week", "Actuals", "segment_id"]
PAT = ["promo", "discount", "holiday", "songkran", "bucha", "festive", "season", "summer", "winter",
       "monsoon", "week_of_year", "is_pre", "is_post", "weeks_to", "stringency", "off_invoice", "baseline", "sp_week", "qtr_cycle"]
FEATS = ["log_vol", "log_mean", "n", "nzfrac", "ADI", "CV2", "spike", "recent"]


def _load(fd, name, keys=None, keep=None):
    """Load a feature panel, filtering to `keys` AT READ TIME (parquet predicate pushdown)
    so the full all-keys panel (~290k rows) never materialises — only the candidate rows."""
    pq = sorted(glob.glob(f"{fd}/{name}*.parquet")); csv = sorted(glob.glob(f"{fd}/{name}*.csv"))
    ks = set(map(str, keys)) if keys is not None else None
    if pq:
        flt = [("key", "in", list(ks))] if ks is not None else None
        try:
            d = pd.read_parquet(pq[0], filters=flt)
        except Exception:
            d = pd.read_parquet(pq[0]); d = d[d["key"].astype(str).isin(ks)] if ks else d
    else:
        d = pd.read_csv(csv[0])
        if ks is not None:
            d = d[d["key"].astype(str).isin(ks)]
    d = d[d["key"].notna()].copy()
    yw = pd.to_numeric(d["year_week"], errors="coerce"); d = d[yw.notna()].copy()
    d["year_week"] = yw[yw.notna()].astype("int64"); d["key"] = d["key"].astype(str)
    idc = [c for c in ID if c in d.columns]
    numc = keep if keep is not None else [c for c in d.columns if c not in idc and pd.api.types.is_numeric_dtype(d[c])]
    valid = [c for c in numc if c in d.columns]; d = d[idc + valid]
    d[valid] = d[valid].replace([np.inf, -np.inf], np.nan); return d, numc


def wape(fc, act):
    m = fc.merge(act, on=["key", "year_week"], how="inner", suffixes=("", "_a"))
    g = m.groupby("key").apply(lambda d: np.abs(d["predicted"] - d["Actuals"]).sum() / max(np.abs(d["Actuals"]).sum(), 1e-9))
    return g


def main():
    t0 = time.time()
    F = pd.read_parquet("artifacts_th_local/_key_features.parquet").set_index("key")
    F["log_vol"] = np.log1p(F["vol"]); F["log_mean"] = np.log1p(F["mean"])
    # stratified candidates: all top-volume + a mid-volume sample spanning the win/lose boundary
    topk = F.nlargest(8 if SMOKE else 180, "vol").index
    mid = F[(F.vol >= F.vol.quantile(.5)) & (F.vol < F.vol.quantile(.85))]
    midk = mid.sample(min(8 if SMOKE else 170, len(mid)), random_state=0).index
    cand = sorted(set(topk) | set(midk))
    print(f"[router] {len(cand)} candidate keys for backtest (smoke={SMOKE})", flush=True)

    # --- pooled deep backtest: LOAD from _router_deep.py output (separate process; see header) ---
    dvf = f"artifacts_th_local/_router_deep_{BT_ORIGIN}.parquet"
    if not glob.glob(dvf):
        print(f"[router] {dvf} missing -> run `python scripts/_router_deep.py {BT_ORIGIN} {400 if SMOKE else 2000}` first")
        return
    deep = pd.read_parquet(dvf)
    print(f"[router] loaded pooled deep backtest: {len(deep)} rows from {dvf}", flush=True)

    # --- per-key backtest + actuals, per category ---
    candset = set(map(str, cand))
    perkey, acts = [], []
    for c in CATS:
        try:
            fd = f"artifacts_th_local/final/{c}/feature_output"
            if not glob.glob(f"{fd}/train_features*"):
                continue
            train, numc = _load(fd, "train_features", keys=candset)   # read-time filter -> only candidate rows
            cc = sorted(set(train["key"].astype(str)))
            if not cc:
                continue
            val, _ = _load(fd, "val_features", keys=candset, keep=numc)
            test, _ = _load(fd, "test_features", keys=candset, keep=numc)
            fwd = [x for x in train.columns if x not in ("key", "year_week", "Actuals") and any(p in x.lower() for p in PAT)]
            train, val, test = hf.augment_panels(train, val, test, "Actuals", fwd)
            yw = pd.to_numeric(train["year_week"], errors="coerce")
            bt_tr, bt_va = train[yw <= 202528], train[yw > 202528]
            print(f"[router] {c}: backtesting {len(cc)} candidates ({time.time()-t0:.0f}s)", flush=True)
            fc = hf.train_perkey_forecasts(bt_tr, bt_va, val, cc, hf.stable_perkey_config(), int(BT_ORIGIN))
            perkey.append(fc[["key", "year_week", "predicted"]])
            acts.append(val[["key", "year_week", "Actuals"]])
            gc.collect()
        except Exception as e:
            print(f"[router] {c} FAILED: {type(e).__name__}: {e}", flush=True)
    if not perkey:
        print("[router] no per-key backtests produced — abort"); return
    perkey = pd.concat(perkey, ignore_index=True); act = pd.concat(acts, ignore_index=True)
    for x in (perkey, deep, act):
        x["key"] = x["key"].astype(str); x["year_week"] = pd.to_numeric(x["year_week"], errors="coerce").astype("int64")

    # --- label: own beats pooled (per-key WAPE < pooled WAPE) on the held-out window ---
    wp = wape(perkey, act); wd = wape(deep, act)
    lab = pd.DataFrame({"wape_perkey": wp, "wape_pooled": wd.reindex(wp.index)}).dropna()
    lab["own"] = (lab["wape_perkey"] < lab["wape_pooled"]).astype(int)
    lab = lab.join(F[FEATS])
    print(f"[router] labels: {lab['own'].sum()}/{len(lab)} keys win with own model", flush=True)

    # --- FFORMS classifier: features -> own/pooled ---
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=0)
    X, y = lab[FEATS].fillna(0), lab["own"]
    if y.nunique() > 1 and len(y) > 20:
        cv = cross_val_score(clf, X, y, cv=4, scoring="accuracy").mean()
        print(f"[router] classifier CV accuracy {cv:.2f}", flush=True)
    clf.fit(X, y)
    imp = sorted(zip(FEATS, clf.feature_importances_), key=lambda z: -z[1])
    print("[router] feature importance:", {k: round(v, 2) for k, v in imp}, flush=True)
    # assign EVERY key (eligibility guard: very low-volume/short keys -> pooled regardless)
    allX = F[FEATS].fillna(0)
    F["own_prob"] = clf.predict_proba(allX)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else 0.0
    F["own"] = ((F["own_prob"] >= 0.5) & (F["vol"] >= F["vol"].quantile(0.5)) & (F["n"] >= 40)).astype(int)
    F[["vol", "own_prob", "own"]].reset_index().to_parquet("artifacts_th_local/_router_assignment.parquet")
    own_vol = F.loc[F.own == 1, "vol"].sum() / F["vol"].sum()
    print(f"[router] ASSIGNMENT: {int(F.own.sum())}/{len(F)} keys -> OWN per-key model "
          f"({own_vol:.0%} of volume); rest -> pooled deep. saved _router_assignment.parquet", flush=True)


if __name__ == "__main__":
    main()
