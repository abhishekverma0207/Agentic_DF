"""Can a TIGHTER (RF-like) per-key recipe match LEGO on FOODS's STABLE keys?
The FABRIC recipe (15 leaves, SPLY-blend 0.4, recency 0) is tuned for NOISY keys;
FOODS high-volume keys are stable, where LEGO's per-key RF has low dispersion.
Test alternative per-key recipes on the top-N FOODS hv keys vs LEGO. Usage: _perkey_recipe.py FOODS 18"""
import os
for k, v in {"MLFLOW_AUTOLOGGING_DISABLE": "1", "OTEL_SDK_DISABLED": "true",
             "CREWAI_DISABLE_TELEMETRY": "true", "CREWAI_DO_NOT_TRACK": "1",
             "DIQ_RUNNER_SKIP_INSTALL": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, time, logging
import numpy as np, pandas as pd
from utils.feature_io import read_features_intermediate
from utils.direct_multihorizon import DirectMHConfig, train_direct_multihorizon, predict_direct_multihorizon
import utils.hybrid_forecast as hf
from utils import lego_eval

CAT = sys.argv[1] if len(sys.argv) > 1 else "FOODS"
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 18
CATNAME = {"FOODS": "FOODS", "FABRIC_CLEANING": "FABRIC CLEANING"}.get(CAT, CAT.replace("_", " "))
ORIGIN = 202602
ID = ["key", "year_week", "Actuals", "segment_id"]


def _load(fd, name, keep=None):
    d = read_features_intermediate(fd, name)
    d = d[d["key"].notna()].copy()
    yw = pd.to_numeric(d["year_week"], errors="coerce")
    d = d[yw.notna()].copy(); d["year_week"] = yw[yw.notna()].astype("int64"); d["key"] = d["key"].astype(str)
    idc = [c for c in ID if c in d.columns]
    numc = keep if keep is not None else [c for c in d.columns if c not in idc and pd.api.types.is_numeric_dtype(d[c])]
    valid = [c for c in numc if c in d.columns]
    d = d[idc + valid]; d[valid] = d[valid].replace([np.inf, -np.inf], np.nan)
    return d, numc


def cfg(objective="regression", recency=0.0, sply=0.4, leaves=15, min_data=8,
        rounds=400, top_k=30, seeds=1):
    return DirectMHConfig(
        key_col="key", time_col="year_week", target_col="Actuals", objective=objective, n_seeds=seeds,
        num_leaves=leaves, min_data_in_leaf=min_data, learning_rate=0.05, lambda_l2=1.0,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, num_boost_round=rounds,
        early_stopping_rounds=40, top_k_features=top_k, recency_halflife_weeks=recency,
        sply_blend_alpha=sply, horizon_workers=1, enable_bias_calibration=True,
        calibration_min=0.5, calibration_max=2.5, calibration_min_samples=8)


RECIPES = {
    "fabric(base)":  cfg(),                                                      # 15lf, sply0.4, rec0
    "deep_nosply":   cfg(recency=26, sply=1.0, leaves=31, min_data=15, rounds=700, top_k=40),
    "deep_sply07":   cfg(recency=26, sply=0.7, leaves=31, min_data=15, rounds=700, top_k=40),
    "auto_nosply":   cfg(objective="auto", recency=26, sply=1.0, leaves=31, min_data=15, rounds=700, top_k=40, seeds=2),
}


def main():
    logging.basicConfig(level=logging.ERROR, stream=sys.stdout, force=True)
    fd = f"artifacts_th_local/{CAT}_hybrid/feature_output"
    assert glob.glob(f"{fd}/train_features*"), f"no panels for {CAT}"
    train, numc = _load(fd, "train_features"); val, _ = _load(fd, "val_features", numc); test, _ = _load(fd, "test_features", numc)
    pat = ["promo", "discount", "holiday", "songkran", "bucha", "festive", "season", "summer", "winter",
           "monsoon", "week_of_year", "is_pre", "is_post", "weeks_to", "stringency", "off_invoice", "baseline", "sp_week", "qtr_cycle"]
    fwd = [c for c in train.columns if c not in ("key", "year_week", "Actuals") and any(p in c.lower() for p in pat)]
    train, val, test = hf.augment_panels(train, val, test, "Actuals", fwd)
    # top-N high-volume stable keys
    hv = hf.select_high_volume_keys(train, vol_quantile=0.65)
    vol = train[train["key"].isin(set(hv))].groupby("key")["Actuals"].sum().sort_values(ascending=False)
    keys = vol.head(TOPN).index.astype(str).tolist()
    print(f"[{CAT}] testing {len(keys)} top stable keys, {len(RECIPES)} recipes", flush=True)

    base = lego_eval.load_eval_base()
    def ev(fc):
        p = lego_eval.attach_forecast(base, fc, pred_col="predicted")
        p = p[(p["Category"] == CATNAME) & (p["cs"].isin({k.split("_")[0] for k in keys}))]
        def ab(s):
            a = s["actual"].values; f = s["ours"].fillna(0).values; t = a.sum()
            return (1 - np.abs(f - a).sum() / t, (f - a).sum() / t)
        a13, b13 = ab(p); p45 = p[p.t.isin([4, 5])]; a45, b45 = ab(p45)
        l13, _ = ab(p.assign(ours=p["lego"])); l45, _ = ab(p45.assign(ours=p45["lego"]))
        return a13, a45, b45, l13, l45

    print(f"  {'recipe':14s} | 13wk acc | t45 acc/bias | LEGO t45 | verdict", flush=True)
    lego45 = None
    for name, c in RECIPES.items():
        t0 = time.time()
        fc = hf.train_perkey_forecasts(train, val, test, keys, c, ORIGIN)
        a13, a45, b45, l13, l45 = ev(fc); lego45 = l45
        v = "WIN" if a45 > l45 else ("close" if a45 > l45 - 0.03 else "")
        print(f"  {name:14s} | {a13:.3f}   | {a45:.3f}/{b45:+.2f}  | {l45:.3f}   | {v}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  LEGO t45 on these {len(keys)} keys = {lego45:.3f}; beat it = crack the high-volume wall", flush=True)


if __name__ == "__main__":
    main()
