"""Hybrid POC on FABRIC_CLEANING (the biggest loser): does a per-key direct-MH
beat the global DMH on high-volume keys, and does the combined hybrid beat LEGO?

Reuses the SANITIZED feature panels from the per-key run (artifacts_th_local/fc_perkey,
E1016 keys) and drives the working DMH directly (no recursive path). __main__-guarded.
"""
import os
for k, v in {"MLFLOW_AUTOLOGGING_DISABLE": "1", "OTEL_SDK_DISABLED": "true",
             "CREWAI_DISABLE_TELEMETRY": "true", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging, time
import numpy as np, pandas as pd
from utils.feature_io import read_features_intermediate
from utils.direct_multihorizon import (DirectMHConfig, train_direct_multihorizon,
                                        predict_direct_multihorizon)
import utils.hybrid_forecast as hf
from utils import lego_eval

ORIGIN = 202602


def ab(df, col):
    a = df["actual"].values; f = df[col].fillna(0).values; s = a.sum()
    return (1 - np.abs(f - a).sum() / s, (f - a).sum() / s) if s > 0 else (np.nan, np.nan)


def score(fc, label, restrict_keys=None):
    base = lego_eval.load_eval_base()
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted")
    p = p[p["Category"] == "FABRIC CLEANING"]
    if restrict_keys is not None:
        cs = {k.split("_")[0] for k in restrict_keys}
        p = p[p["cs"].isin(cs)]
    a13, b13 = ab(p, "ours"); a45, b45 = ab(p[p.t.isin([4, 5])], "ours")
    l13, _ = ab(p, "lego"); l45, _ = ab(p[p.t.isin([4, 5])], "lego")
    print(f"  {label:28s} 13wk {a13:6.3f}/{b13:+.2f} | t45 {a45:6.3f}/{b45:+.2f}  (LEGO {l13:.3f}/{l45:.3f})", flush=True)
    return a45


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout, force=True,
                        format="%(asctime)s %(levelname).1s | %(message)s")
    fd = "artifacts_th_local/fc_perkey/feature_output"
    ID = ["key", "year_week", "Actuals", "segment_id"]
    def _load(name, keep_cols=None):
        d = read_features_intermediate(fd, name)
        d = d[d["key"].notna()].copy()
        yw = pd.to_numeric(d["year_week"], errors="coerce")
        d = d[yw.notna()].copy()
        d["year_week"] = yw[yw.notna()].astype("int64")
        d["key"] = d["key"].astype(str)
        # keep ids + NUMERIC features only (raw string cols crash LightGBM); strip inf
        idc = [c for c in ID if c in d.columns]
        numc = keep_cols if keep_cols is not None else [
            c for c in d.columns if c not in idc and pd.api.types.is_numeric_dtype(d[c])]
        valid = [c for c in numc if c in d.columns]
        d = d[idc + valid]
        d[valid] = d[valid].replace([np.inf, -np.inf], np.nan)
        return d, numc
    train, numc = _load("train_features")
    val, _ = _load("val_features", keep_cols=numc)
    test, _ = _load("test_features", keep_cols=numc)
    print(f"[poc] kept {len(numc)} numeric features", flush=True)
    print(f"[poc] panels: train={train.shape} val={val.shape} test={test.shape} | keys={train['key'].nunique()}", flush=True)

    # ---- forward-feature augmentation (trajectory + target-week lead-4 = our edge) ----
    pat = ["promo", "discount", "holiday", "songkran", "bucha", "festive", "season",
           "summer", "winter", "monsoon", "week_of_year", "is_pre", "is_post",
           "weeks_to", "stringency", "off_invoice", "baseline", "sp_week", "qtr_cycle"]
    fwd_cols = [c for c in train.columns if c not in ("key", "year_week", "Actuals")
                and any(p in c.lower() for p in pat)]
    print(f"[poc] forward candidate cols: {len(fwd_cols)}", flush=True)
    train, val, test = hf.augment_panels(train, val, test, target_col="Actuals", fwd_cols=fwd_cols)
    print(f"[poc] augmented panels: train={train.shape}", flush=True)

    # ---- GLOBAL DMH over all E1016 keys ----
    gcfg = DirectMHConfig(key_col="key", time_col="year_week", target_col="Actuals",
                          objective="auto", n_seeds=1, num_leaves=63, min_data_in_leaf=200,
                          num_boost_round=1500, early_stopping_rounds=120, top_k_features=50,
                          recency_halflife_weeks=26, horizon_workers=2)
    t0 = time.time()
    arts_g = train_direct_multihorizon(train, val, gcfg)
    global_fc = predict_direct_multihorizon(test, arts_g, gcfg, origin_week=ORIGIN, fallback_features=val)
    print(f"[poc] global DMH done in {time.time()-t0:.0f}s -> {len(global_fc)} rows", flush=True)

    # ---- PER-KEY DMH for high-volume keys ----
    hv = hf.select_high_volume_keys(train, target_col="Actuals", vol_quantile=0.60)
    print(f"[poc] high-volume keys: {len(hv)}", flush=True)
    t0 = time.time()
    perkey_fc = hf.train_perkey_forecasts(train, val, test, hv,
                                          hf.perkey_dmh_config(num_boost_round=400), ORIGIN)
    print(f"[poc] per-key DMH done in {time.time()-t0:.0f}s -> {len(perkey_fc)} rows", flush=True)

    print("\n[poc] FABRIC CLEANING — per-key vs global (no fwd-aug POC):", flush=True)
    print(">>> on HIGH-VOLUME keys only (where per-key should win):", flush=True)
    score(global_fc, "global (hv subset)", restrict_keys=hv)
    score(perkey_fc, "per-key (hv subset)", restrict_keys=hv)
    print(">>> on ALL FABRIC_CLEANING E1016 keys:", flush=True)
    score(global_fc, "global (all)")
    global_fc.to_parquet("artifacts_th_local/_poc_global.parquet", index=False)
    perkey_fc.to_parquet("artifacts_th_local/_poc_perkey.parquet", index=False)
    print(">>> on HIGH-VOLUME keys (per-key+fwd vs global+fwd):", flush=True)
    score(global_fc, "global+fwd (hv subset)", restrict_keys=hv)
    score(perkey_fc, "per-key+fwd (hv subset)", restrict_keys=hv)
    for blend in [1.0, 0.7]:
        hyb = hf.combine(global_fc, perkey_fc, hv, blend=blend)
        score(hyb, f"HYBRID+fwd (blend={blend})")


if __name__ == "__main__":
    main()
