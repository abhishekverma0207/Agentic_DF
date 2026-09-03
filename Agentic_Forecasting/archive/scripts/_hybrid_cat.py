"""Generalized hybrid for one category (E1016 keys): generate panels via the
pipeline, then global DMH (sparse) + per-key direct-MH (high-volume, locked recipe),
combine, score vs LEGO. Usage: _hybrid_cat.py FOODS   (basis for runner integration)."""
import os
for k, v in {"MLFLOW_AUTOLOGGING_DISABLE": "1", "OTEL_SDK_DISABLED": "true",
             "CREWAI_DISABLE_TELEMETRY": "true", "CREWAI_DO_NOT_TRACK": "1",
             "DIQ_RUNNER_SKIP_INSTALL": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, time, logging
import numpy as np, pandas as pd
from config.schema import load_config_from_yaml
from utils.deterministic_pipeline import run_full_deterministic_pipeline
from utils.feature_io import read_features_intermediate
from utils.direct_multihorizon import DirectMHConfig, train_direct_multihorizon, predict_direct_multihorizon
import utils.hybrid_forecast as hf
from utils import lego_eval

CAT = sys.argv[1] if len(sys.argv) > 1 else "FOODS"
CATNAME = {"FOODS": "FOODS", "FABRIC_CLEANING": "FABRIC CLEANING", "FACE": "FACE",
           "HOME_AND_HYGIENE": "HOME & HYGIENE",
           "DEODORANTS_AND_FRAGRANCES": "DEODORANTS & FRAGRANCES"}.get(CAT, CAT.replace("_", " "))
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


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout, force=True,
                        format="%(asctime)s %(levelname).1s | %(message)s")
    art = f"artifacts_th_local/{CAT}_hybrid"
    feat_dir = f"{art}/feature_output"
    # 1-2. generate panels via the pipeline (E1016 only) if not present
    if not glob.glob(f"{feat_dir}/train_features*"):
        df = pd.read_parquet(f"sourcedata/THAILAND/by_category/{CAT}.parquet")
        e16 = df[df["key"].astype(str).str.endswith("_E1016")].copy()
        inp = f"sourcedata/THAILAND/by_category/_{CAT}_E1016.parquet"
        e16.to_parquet(inp, index=False)
        print(f"[{CAT}] E1016: {e16['key'].nunique()} keys; running pipeline for panels...", flush=True)
        cfg = load_config_from_yaml("config/config_th_base_local.yaml")
        cfg.input_data_path = inp; cfg.artifact_base_path = art; cfg.output_folder_path = art
        cfg.design.dmh_n_seeds = 1
        run_full_deterministic_pipeline(cfg, clean_artifacts=True)
    # 3. load + augment
    train, numc = _load(feat_dir, "train_features"); val, _ = _load(feat_dir, "val_features", numc)
    test, _ = _load(feat_dir, "test_features", numc)
    pat = ["promo", "discount", "holiday", "songkran", "bucha", "festive", "season", "summer", "winter",
           "monsoon", "week_of_year", "is_pre", "is_post", "weeks_to", "stringency", "off_invoice", "baseline", "sp_week", "qtr_cycle"]
    fwd = [c for c in train.columns if c not in ("key", "year_week", "Actuals") and any(p in c.lower() for p in pat)]
    train, val, test = hf.augment_panels(train, val, test, "Actuals", fwd)
    print(f"[{CAT}] panels train={train.shape} keys={train['key'].nunique()} fwd={len(fwd)}", flush=True)
    # 4. global DMH
    gcfg = DirectMHConfig(key_col="key", time_col="year_week", target_col="Actuals", objective="auto",
                          n_seeds=1, num_leaves=63, min_data_in_leaf=200, num_boost_round=1500,
                          early_stopping_rounds=120, top_k_features=50, recency_halflife_weeks=26, horizon_workers=2)
    t0 = time.time()
    arts_g = train_direct_multihorizon(train, val, gcfg)
    global_fc = predict_direct_multihorizon(test, arts_g, gcfg, origin_week=ORIGIN, fallback_features=val)
    print(f"[{CAT}] global DMH {time.time()-t0:.0f}s", flush=True)
    # 5. OOS recipe-adaptive selection: each hv key -> {global, noisy, stable} by 2025-Q1 contest
    from collections import Counter
    hv = hf.select_high_volume_keys(train, vol_quantile=0.65)
    recipes = hf.default_recipes()
    t0 = time.time()
    blind = os.environ.get("HYBRID_BLIND")  # 'noisy'|'stable' -> skip OOS, all hv use that recipe
    if blind in ("noisy", "stable"):
        assignment, seltbl = {str(k): blind for k in hv}, pd.DataFrame()
        print(f"[{CAT}] BLIND mode: all {len(hv)} hv keys -> '{blind}' recipe (no OOS selection)", flush=True)
    else:
        assignment, seltbl = hf.oos_select(train, val, gcfg, recipes, hv)
    selected = list(assignment.keys())
    print(f"[{CAT}] OOS-select {time.time()-t0:.0f}s: {len(selected)}/{len(hv)} hv keys -> per-key; "
          f"recipes={dict(Counter(assignment.values()))}", flush=True)
    if not seltbl.empty:
        seltbl.to_parquet(f"{art}/_oos_select.parquet", index=False)
    t0 = time.time()
    perkey_fc = hf.train_perkey_assigned(train, val, test, assignment, recipes, ORIGIN)
    print(f"[{CAT}] per-key DMH {time.time()-t0:.0f}s ({len(selected)} keys)", flush=True)
    # 6. combine + eval
    base = lego_eval.load_eval_base()
    def ev(fc, tag, keys=None):
        fc = fc.copy()
        p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p["Category"] == CATNAME]
        if keys is not None:
            p = p[p["cs"].isin({k.split("_")[0] for k in keys})]
        def ab(s):
            a = s["actual"].values; f = s["ours"].fillna(0).values; t = a.sum(); return (1 - np.abs(f - a).sum() / t, (f - a).sum() / t)
        a13, b13 = ab(p); a45, b45 = ab(p[p.t.isin([4, 5])]); l13, _ = ab(p.assign(ours=p["lego"])); l45, _ = ab(p[p.t.isin([4, 5])].assign(ours=p["lego"]))
        win = a45 > l45
        print(f"  {tag:22s} 13wk {a13:.3f}/{b13:+.2f} | t45 {a45:.3f}/{b45:+.2f}  (LEGO {l13:.3f}/{l45:.3f}) {'<-- WIN acc' if win else ''}", flush=True)
    print(f"\n[{CAT}] HYBRID-OOS vs LEGO:", flush=True)
    ev(global_fc, "global+fwd", keys=hv)
    ev(global_fc, "global (all)")
    if selected:
        ev(perkey_fc, "per-key (sel)", keys=selected)
        hyb = hf.combine(global_fc, perkey_fc, selected, blend=1.0)
        ev(hyb, "HYBRID-OOS (all)")
        hyb.to_parquet(f"{art}/_hybrid_oos_fc.parquet", index=False)
    else:
        print("  [no keys selected -> HYBRID-OOS == global (all)]", flush=True)
        global_fc.to_parquet(f"{art}/_hybrid_oos_fc.parquet", index=False)


if __name__ == "__main__":
    main()
