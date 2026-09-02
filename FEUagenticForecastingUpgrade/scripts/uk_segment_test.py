#!/usr/bin/env python
"""
Test the segment×category×LEGO ensemble for UK using the PRODUCTION utils.segment_engine
(also confirms it's market-generic). UK mirrors DE: panel 01_all_keys_dr_adjusted ⋈ FS
(FS carries lego_segment), benchmark TF_pristine_allkeys/prediction_rf, actuals = 2026-25
panel actual_sales. Reads UC directly (off-cluster); caches the benchmark + components to /tmp.

    .venv/bin/python scripts/uk_segment_test.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace
import numpy as np, pandas as pd, yaml
from utils import uc_io
from utils import segment_engine as se

SNAP = "/Volumes/pds_feu_931272_dev/eu_uk/landing/UK_PROD_NEW/lego_runs/forward_run/total_forecast/snapshot"
CONFIG = "config/config_uk_base_local.yaml"
KEY, WEEK, TARGET, CAT, SEG = "key", "year_week", "actual_sales", "category_name", "lego_segment"
FIT_SNAPS = ["2026-09", "2026-11"]
EVAL_SNAP = "2026-13"
ACTUALS_SNAP = "2026-25"
H, LAGS = 5, [4, 5]
INSCOPE = {"BEVERAGE", "CONDIMENT", "COOKING AID", "DEODORANT & FRAGRANCE", "FABRIC CLEANING",
           "FABRIC ENHANCER", "HAIR CARE", "HEALTH & WELLBEING", "HOME & HYGIENE", "MINI MEAL",
           "ORAL CARE", "OTH FOOD", "SKIN CARE", "SKIN CLEANSING"}


def _acc(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return 1.0 - np.abs(f - a).sum() / max(a.sum(), 1e-9)


def _bias(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return (f - a).sum() / max(a.sum(), 1e-9)


def panel_keep(schema):
    from utils.config_loader import load_config_file
    cfg = load_config_file(CONFIG)          # prefers config_uk_base_local.py, falls back to .yaml
    ids = [KEY, WEEK, "cs_gtin", CAT, TARGET]
    explicit = ids + cfg.get("cat_features", []) + cfg.get("extra_signal_cols", [])
    pref = tuple(cfg.get("future_cov_prefixes", []))
    s = set(schema)
    keep = [c for c in explicit if c in s]
    keep += [c for c in schema if c not in keep and c.startswith(pref)]
    return list(dict.fromkeys(keep))


def read_full(snap):
    pan = uc_io.read_parquet(f"{SNAP}/{snap}/01_all_keys_dr_adjusted", max_parts=1)
    keep = panel_keep(pan.columns.tolist())
    pan = uc_io.read_parquet(f"{SNAP}/{snap}/01_all_keys_dr_adjusted", columns=keep, max_workers=16)
    fs0 = uc_io.read_parquet(f"{SNAP}/{snap}/FS", max_parts=1)
    fcols = [KEY, WEEK] + [c for c in fs0.columns if c == SEG or c.startswith("f_")]
    fs = uc_io.read_parquet(f"{SNAP}/{snap}/FS", columns=fcols, max_workers=16)
    pan[KEY] = pan[KEY].astype(str); fs[KEY] = fs[KEY].astype(str)
    add = [c for c in fs.columns if c in (KEY, WEEK) or c not in pan.columns]
    full = pan.merge(fs[add], on=[KEY, WEEK], how="inner")
    full = full[full[CAT].isin(INSCOPE)].copy()
    print(f"  [{snap}] panel⋈FS {full.shape}, keys={full[KEY].nunique()}, segs={full[SEG].nunique()}, cats={full[CAT].nunique()}")
    return full


def components(snap):
    cpath = f"/tmp/uk_seg_comp_{snap}.parquet"
    if os.path.exists(cpath):
        return pd.read_parquet(cpath)
    full = read_full(snap)
    bench = uc_io.read_parquet(f"{SNAP}/{snap}/TF_pristine_allkeys",
                               columns=[KEY, WEEK, "prediction_rf", SEG], max_workers=16)
    bp = f"/tmp/uk_bench_{snap}.parquet"; bench.to_parquet(bp, index=False)
    comp = se.compute_segment_components(
        SimpleNamespace(forecast_horizon=H), full, snapshot_week=snap, route_categories=sorted(INSCOPE),
        segment_col=SEG, lego_source=bp, config_path=CONFIG, lego_pred_col="prediction_rf", horizons=H)
    comp = comp[comp["horizon"].isin(LAGS)]
    comp.to_parquet(cpath, index=False)
    return comp


def main():
    acts = uc_io.read_parquet(f"{SNAP}/{ACTUALS_SNAP}/01_all_keys_dr_adjusted",
                              columns=[KEY, WEEK, TARGET], max_workers=16)
    acts[KEY] = acts[KEY].astype(str)
    acts[WEEK] = acts[WEEK].astype(str).str.replace("-", "", regex=False)
    acts = acts.rename(columns={TARGET: "actual"}).groupby([KEY, WEEK], as_index=False)["actual"].sum()

    comps = {}
    for snap in FIT_SNAPS + [EVAL_SNAP]:
        c = components(snap)
        c[KEY] = c[KEY].astype(str)
        comps[snap] = c.merge(acts, on=[KEY, WEEK], how="inner")
        print(f"  [{snap}] {len(comps[snap])} component rows joined to actuals")

    fitted = se.fit_segment_weights(pd.concat([comps[s] for s in FIT_SNAPS], ignore_index=True))
    ev = comps[EVAL_SNAP].copy()
    ev["blend"] = se.blend_with_weights(ev, {k: v["weights"] for k, v in fitted.items()})

    def sc(fr, col):
        g = fr.groupby(["cs_gtin", WEEK]).agg(p=(col, "sum"), a=("actual", "sum")).reset_index()
        return _acc(g.p, g.a), _bias(g.p, g.a)

    print(f"\n===== UK segment×category×LEGO  (fit {FIT_SNAPS} -> eval {EVAL_SNAP}, GTIN grain, lags {LAGS}) =====")
    print(f"{'category':<26}{'acc_LEGO':>9}{'acc_BLEND':>10}{'Δacc':>8}  {'bias_LEGO':>10}{'bias_BLEND':>11}  win")
    nwin = 0
    for cat in sorted(ev["category"].dropna().unique()):
        s = ev[ev["category"] == cat]
        aL, bL = sc(s, "rf"); aB, bB = sc(s, "blend")
        win = (aB >= aL - 1e-3) and (min(bB, 0) >= min(bL, 0) - 0.02); nwin += win
        print(f"{cat:<26}{aL:>9.3f}{aB:>10.3f}{aB-aL:>+8.3f}  {bL:>+10.3f}{bB:>+11.3f}  {'✓' if win else ''}")
    aL, bL = sc(ev, "rf"); aB, bB = sc(ev, "blend")
    print(f"{'— ALL (pooled) —':<26}{aL:>9.3f}{aB:>10.3f}{aB-aL:>+8.3f}  {bL:>+10.3f}{bB:>+11.3f}")
    nstk = sum(1 for v in fitted.values() if v["use"] == "STACK")
    print(f"\nBLEND beats LEGO: {nwin}/{ev['category'].nunique()} categories | STACK deployed {nstk}/{len(fitted)} segments")


if __name__ == "__main__":
    main()
