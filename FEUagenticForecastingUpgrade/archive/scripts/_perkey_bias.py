"""Chase the high-volume bias: grid per-key (objective, recency, sply_blend) on the
FABRIC_CLEANING high-volume keys, combine with the saved global, eval vs LEGO.
Goal: t45 acc >= 0.278 AND |bias| < 0.11 (beat LEGO on BOTH)."""
import os
for k, v in {"MLFLOW_AUTOLOGGING_DISABLE": "1", "OTEL_SDK_DISABLED": "true",
             "CREWAI_DISABLE_TELEMETRY": "true", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import numpy as np, pandas as pd
from utils.feature_io import read_features_intermediate
from utils.direct_multihorizon import DirectMHConfig
import utils.hybrid_forecast as hf
from utils import lego_eval

ORIGIN = 202602
ID = ["key", "year_week", "Actuals", "segment_id"]


def _load(fd, name, keep=None):
    d = read_features_intermediate(fd, name)
    d = d[d["key"].notna()].copy()
    yw = pd.to_numeric(d["year_week"], errors="coerce")
    d = d[yw.notna()].copy()
    d["year_week"] = yw[yw.notna()].astype("int64"); d["key"] = d["key"].astype(str)
    idc = [c for c in ID if c in d.columns]
    numc = keep if keep is not None else [c for c in d.columns if c not in idc and pd.api.types.is_numeric_dtype(d[c])]
    valid = [c for c in numc if c in d.columns]
    d = d[idc + valid]; d[valid] = d[valid].replace([np.inf, -np.inf], np.nan)
    return d, numc


def cfg(obj, rec, sply):
    return DirectMHConfig(key_col="key", time_col="year_week", target_col="Actuals",
                          objective=obj, n_seeds=1, num_leaves=15, min_data_in_leaf=8,
                          num_boost_round=400, early_stopping_rounds=40, top_k_features=30,
                          recency_halflife_weeks=rec, sply_blend_alpha=sply, horizon_workers=1,
                          enable_bias_calibration=True, calibration_min=0.5, calibration_max=2.5,
                          calibration_min_samples=8)


def main():
    logging.basicConfig(level=logging.ERROR, stream=sys.stdout, force=True)
    fd = "artifacts_th_local/fc_perkey/feature_output"
    train, numc = _load(fd, "train_features")
    val, _ = _load(fd, "val_features", numc)
    test, _ = _load(fd, "test_features", numc)
    pat = ["promo", "discount", "holiday", "songkran", "bucha", "festive", "season", "summer",
           "winter", "monsoon", "week_of_year", "is_pre", "is_post", "weeks_to", "stringency",
           "off_invoice", "baseline", "sp_week", "qtr_cycle"]
    fwd = [c for c in train.columns if c not in ("key", "year_week", "Actuals") and any(p in c.lower() for p in pat)]
    train, val, test = hf.augment_panels(train, val, test, "Actuals", fwd)
    sply_cols = [c for c in train.columns if "lag_52" in c.lower() or "lag52" in c.lower()]
    print(f"[bias] augmented train={train.shape}; lag_52 cols for SPLY blend: {sply_cols[:3]}", flush=True)

    g = pd.read_parquet("artifacts_th_local/_poc_global.parquet")
    hv = hf.select_high_volume_keys(train, vol_quantile=0.65)
    print(f"[bias] high-volume keys: {len(hv)}", flush=True)
    base = lego_eval.load_eval_base()

    def ev(fc, tag):
        fc = fc.copy(); fc["Category"] = "FABRIC CLEANING"
        p = lego_eval.attach_forecast(base, fc, pred_col="predicted")
        p = p[p["Category"] == "FABRIC CLEANING"]
        def ab(s):
            a = s["actual"].values; f = s["ours"].fillna(0).values; t = a.sum()
            return (1 - np.abs(f - a).sum() / t, (f - a).sum() / t)
        a13, b13 = ab(p); a45, b45 = ab(p[p.t.isin([4, 5])])
        win = (a45 > 0.278) and (abs(b45) < 0.11)
        print(f"  {tag:28s} 13wk {a13:.3f}/{b13:+.2f} | t45 {a45:.3f}/{b45:+.2f}  {'<<< BEATS LEGO BOTH' if win else ''}", flush=True)

    print("LEGO t45 0.278/-0.11; current hybrid t45 0.279/-0.17", flush=True)
    variants = [
        ("regression rec26", cfg("regression", 26, 1.0)),
        ("regression rec0", cfg("regression", 0, 1.0)),
        ("auto rec0", cfg("auto", 0, 1.0)),
        ("regression rec0 sply.6", cfg("regression", 0, 0.6)),
        ("auto rec0 sply.6", cfg("auto", 0, 0.6)),
    ]
    for label, c in variants:
        pk = hf.train_perkey_forecasts(train, val, test, hv, c, ORIGIN)
        ev(hf.combine(g, pk, hv, blend=1.0), label)


if __name__ == "__main__":
    main()
