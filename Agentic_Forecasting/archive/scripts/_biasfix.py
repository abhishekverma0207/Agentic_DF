"""Does an OOS-selected bias/seasonal correction close the high-volume gap to LEGO?
Trains the global (full) -> test forecast, AND a reduced global -> 2025-Q1 OOS forecast.
For each candidate correction, reports OOS accuracy (the SELECTABLE signal) and the
true TEST accuracy vs LEGO. If best-by-OOS == best-on-test, OOS selection of the
correction is justified. Usage: _biasfix.py FOODS"""
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
CATNAME = {"FOODS": "FOODS", "FABRIC_CLEANING": "FABRIC CLEANING", "FACE": "FACE",
           "HOME_AND_HYGIENE": "HOME & HYGIENE"}.get(CAT, CAT.replace("_", " "))
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


def _sply_week(yw):  # same ISO week, last year
    return (yw // 100 - 1) * 100 + yw % 100


def hist_actuals(train, val):
    h = pd.concat([train[["key", "year_week", "Actuals"]], val[["key", "year_week", "Actuals"]]], ignore_index=True)
    h["year_week"] = h["year_week"].astype(int)
    return h.groupby(["key", "year_week"])["Actuals"].sum().reset_index()


def apply_correction(fc, hist, mode, origin, lo=0.5, hi=3.0):
    """Scale fc['predicted'] per key to a target level. mode in
    {none, sply, yoy}. Reliability-gated by SPLY presence."""
    f = fc.copy(); f["key"] = f["key"].astype(str); f["year_week"] = f["year_week"].astype(int)
    if mode == "none":
        return f
    hk = hist.set_index(["key", "year_week"])["Actuals"]
    # per-key recent YoY growth (last 8 wk this yr / same 8 wk last yr)
    recent_wks = [origin - i for i in range(8)]
    ly_wks = [_sply_week(w) for w in recent_wks]
    g = f.groupby("key")
    scales = {}
    for key, grp in g:
        tw = grp["year_week"].tolist()
        sply_tot = sum(float(hk.get((key, _sply_week(w)), np.nan)) if (key, _sply_week(w)) in hk.index else np.nan for w in tw)
        sply_tot = np.nansum([hk.get((key, _sply_week(w)), np.nan) for w in tw])
        pred_tot = float(grp["predicted"].sum())
        present = np.mean([(key, _sply_week(w)) in hk.index for w in tw])  # SPLY reliability
        if pred_tot <= 0 or sply_tot <= 0 or present <= 0:
            scales[key] = 1.0; continue
        target = sply_tot
        if mode == "yoy":
            rt = np.nansum([hk.get((key, w), np.nan) for w in recent_wks])
            lt = np.nansum([hk.get((key, w), np.nan) for w in ly_wks])
            growth = np.clip(rt / lt, 0.5, 2.0) if lt > 0 else 1.0
            target = sply_tot * growth
        raw_scale = np.clip(target / pred_tot, lo, hi)
        scales[key] = 1.0 + present * (raw_scale - 1.0)  # shrink toward 1 by reliability
    f["predicted"] = f["predicted"] * f["key"].map(scales).fillna(1.0)
    return f


def main():
    logging.basicConfig(level=logging.ERROR, stream=sys.stdout, force=True)
    art = f"artifacts_th_local/{CAT}_hybrid"; fd = f"{art}/feature_output"
    assert glob.glob(f"{fd}/train_features*"), f"no panels for {CAT}; run _hybrid_cat first"
    train, numc = _load(fd, "train_features"); val, _ = _load(fd, "val_features", numc); test, _ = _load(fd, "test_features", numc)
    pat = ["promo", "discount", "holiday", "songkran", "bucha", "festive", "season", "summer", "winter",
           "monsoon", "week_of_year", "is_pre", "is_post", "weeks_to", "stringency", "off_invoice", "baseline", "sp_week", "qtr_cycle"]
    fwd = [c for c in train.columns if c not in ("key", "year_week", "Actuals") and any(p in c.lower() for p in pat)]
    train, val, test = hf.augment_panels(train, val, test, "Actuals", fwd)
    gcfg = DirectMHConfig(key_col="key", time_col="year_week", target_col="Actuals", objective="auto",
                          n_seeds=1, num_leaves=63, min_data_in_leaf=200, num_boost_round=1500,
                          early_stopping_rounds=120, top_k_features=50, recency_halflife_weeks=26, horizon_workers=2)
    hist = hist_actuals(train, val)
    yw = pd.to_numeric(train["year_week"], errors="coerce")
    tr_sel = train[yw <= 202450]; vl_sel = train[(yw > 202450) & (yw <= 202502)]; oos = train[(yw > 202502) & (yw <= 202515)]
    hist_oos = hist_actuals(tr_sel, vl_sel)  # history available as-of the OOS origin only
    oos_act = oos[["key", "year_week", "Actuals"]].copy(); oos_act["year_week"] = oos_act["year_week"].astype(int)
    tcache, ocache = f"{art}/_bf_testfc.parquet", f"{art}/_bf_oosfc.parquet"
    if glob.glob(tcache) and glob.glob(ocache):
        test_fc = pd.read_parquet(tcache); oos_fc = pd.read_parquet(ocache)
        print(f"[{CAT}] loaded cached global forecasts", flush=True)
    else:
        t0 = time.time(); arts = train_direct_multihorizon(train, val, gcfg)
        test_fc = predict_direct_multihorizon(test, arts, gcfg, origin_week=ORIGIN, fallback_features=val)
        arts_r = train_direct_multihorizon(tr_sel, vl_sel, gcfg)
        oos_fc = predict_direct_multihorizon(oos, arts_r, gcfg, origin_week=202502, fallback_features=vl_sel)
        test_fc.to_parquet(tcache, index=False); oos_fc.to_parquet(ocache, index=False)
        print(f"[{CAT}] globals trained {time.time()-t0:.0f}s", flush=True)
    # category-level OOS-learned uplift scalars (the robust, low-variance bias fix)
    mo = oos_fc.copy(); mo["key"] = mo["key"].astype(str); mo["year_week"] = mo["year_week"].astype(int)
    mo = mo.merge(oos_act, on=["key", "year_week"], how="inner", suffixes=("", "_a"))
    aa = mo["Actuals_a"] if "Actuals_a" in mo else mo["Actuals"]
    scal = float(aa.sum() / mo["predicted"].sum()) if mo["predicted"].sum() > 0 else 1.0
    mo45 = mo[mo["horizon"].isin([4, 5])] if "horizon" in mo else mo
    aa45 = mo45["Actuals_a"] if "Actuals_a" in mo45 else mo45["Actuals"]
    scal45 = float(aa45.sum() / mo45["predicted"].sum()) if mo45["predicted"].sum() > 0 else 1.0
    print(f"[{CAT}] OOS-learned uplift: overall x{scal:.3f}, t45 x{scal45:.3f}", flush=True)

    def oos_acc(fc):
        m = fc.copy(); m["key"] = m["key"].astype(str); m["year_week"] = m["year_week"].astype(int)
        m = m.merge(oos_act, on=["key", "year_week"], how="inner", suffixes=("", "_a"))
        a = m["Actuals_a"].values if "Actuals_a" in m else m["Actuals"].values
        f = m["predicted"].values; t = np.abs(a).sum()
        m45 = m[m["horizon"].isin([4, 5])] if "horizon" in m else m
        a4 = m45.get("Actuals_a", m45.get("Actuals")).values; f4 = m45["predicted"].values; t4 = np.abs(a4).sum()
        return (1 - np.abs(f - a).sum() / t if t > 0 else np.nan,
                1 - np.abs(f4 - a4).sum() / t4 if t4 > 0 else np.nan,
                (f4 - a4).sum() / t4 if t4 > 0 else np.nan)

    base = lego_eval.load_eval_base()
    def test_acc(fc):
        p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p["Category"] == CATNAME]
        def ab(s):
            a = s["actual"].values; f = s["ours"].fillna(0).values; t = a.sum(); return (1 - np.abs(f - a).sum() / t, (f - a).sum() / t)
        a13, b13 = ab(p); p45 = p[p.t.isin([4, 5])]; a45, b45 = ab(p45)
        l13, _ = ab(p.assign(ours=p["lego"])); l45, _ = ab(p45.assign(ours=p45["lego"]))
        return a13, a45, b45, l13, l45

    print(f"\n[{CAT}] CORRECTION: OOS-acc (selectable) vs TEST-acc (truth) vs LEGO", flush=True)
    print(f"  {'mode':10s} | OOS t45acc | TEST 13/t45 acc  bias | LEGO 13/t45", flush=True)
    for mode, sc in [("none", None), ("cat", scal), ("cat45", scal45), ("sply", None), ("yoy", None)]:
        if sc is not None:  # uniform OOS-learned scalar (OOS-acc is circular -> not shown)
            tc = test_fc.copy(); tc["predicted"] = tc["predicted"] * sc; ostr = "  -  "
        else:
            tc = apply_correction(test_fc, hist, mode, ORIGIN)
            _, o45, _ = oos_acc(apply_correction(oos_fc, hist_oos, mode, 202502)); ostr = f"{o45:.3f}"
        a13, a45, b45, l13, l45 = test_acc(tc)
        win = "<-- WIN t45" if a45 > l45 else ""
        print(f"  {mode:10s} | {ostr}     | {a13:.3f}/{a45:.3f}  {b45:+.2f} | {l13:.3f}/{l45:.3f} {win}", flush=True)


if __name__ == "__main__":
    main()
