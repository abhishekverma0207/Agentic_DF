#!/usr/bin/env python3
"""
Per-origin consistency check: for each of the 7 rolling origins, does the
rf-ensemble (global+percat+LEGO blend, weights learned leave-one-origin-out)
beat LEGO on BOTH metrics, per category? We want the 7/10 winners to hold on
every single origin, not just pooled.

Generates the 3 missing base forecasts (202610/12/14) at full strength on the
TRIM cache (matches the existing experiments_acc/ artifacts), then per-origin
leave-one-origin-out scoring at GTIN lag-4.
"""
from __future__ import annotations
import os, sys, warnings, logging, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import uk_eval as E, uk_forecast as F

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SN = ["202609", "202610", "202611", "202612", "202613", "202614", "202615"]
SRC = "artifacts_uk_local/experiments_acc"
OUT = "artifacts_uk_local/per_origin"
os.makedirs(OUT, exist_ok=True)

def base_for_all_origins():
    """Return (global_df, percat_df) with all 7 origins. Reuse saved (4) + generate (3)."""
    g_path, l_path = f"{OUT}/global_7.parquet", f"{OUT}/percat_7.parquet"
    if os.path.exists(g_path) and os.path.exists(l_path):
        return pd.read_parquet(g_path), pd.read_parquet(l_path)
    g_old = pd.read_parquet(f"{SRC}/forecasts_global.parquet"); g_old["snapshot"]=g_old["snapshot"].astype(str)
    l_old = pd.read_parquet(f"{SRC}/forecasts_percat_all.parquet"); l_old["snapshot"]=l_old["snapshot"].astype(str)
    have = set(g_old["snapshot"].unique())
    missing = [s for s in SN if s not in have]
    print(f"  have {sorted(have)} | missing {missing}", flush=True)
    g_new, l_new = [], []
    for s in missing:
        t0 = time.time()
        cfg_g = F.load_config(overrides={"per_category": False, "bias_calibration": False})
        cfg_l = F.load_config(overrides={"per_category": True,  "bias_calibration": False})
        g_new.append(F.forecast_origin(s, cfg_g, horizons=[4]).assign(snapshot=s))
        l_new.append(F.forecast_origin(s, cfg_l, horizons=[4]).assign(snapshot=s))
        print(f"  origin {s} bases generated in {time.time()-t0:.0f}s", flush=True)
    g = pd.concat([g_old, *g_new], ignore_index=True); g = g[g["snapshot"].isin(SN)]
    l = pd.concat([l_old, *l_new], ignore_index=True); l = l[l["snapshot"].isin(SN)]
    g.to_parquet(g_path, index=False); l.to_parquet(l_path, index=False)
    return g, l


def main():
    print("Generating bases for all 7 origins…", flush=True)
    g, l = base_for_all_origins()
    fr = E.load_scoring_frame(SN, lags=(4,))
    fr = E.attach_candidate(fr, g, value_col="forecast", name="g")
    fr = E.attach_candidate(fr, l, value_col="forecast", name="l")
    fr["g"] = fr["g"].fillna(0); fr["l"] = fr["l"].fillna(0)
    # gtin grain
    gt = fr.groupby(["category_name","cs_gtin","year_week","horizon","snapshot"], as_index=False).agg(
        g=("g","sum"), l=("l","sum"), rf=("prediction_rf","sum"), actual=("actual","sum"))

    # Simplex over (wg, wl, wrf) at step 0.1 — 66 candidates
    simplex = [(wg, wl, round(1-wg-wl, 2))
               for wg in np.arange(0, 1.01, 0.1)
               for wl in np.arange(0, 1.01-wg, 0.1)]

    def wape(f, A, sa): return np.abs(f-A).sum()/sa

    # Leave-one-origin-out: weights from the other 6 origins, applied to held-out
    parts = []
    for test in SN:
        tr = gt[gt.snapshot != test]
        te = gt[gt.snapshot == test].copy()
        wmap = {}
        for cat, cc in tr.groupby("category_name"):
            A = cc.actual.values; sa = A.sum()
            if sa <= 0: wmap[cat] = (0, 0, 1); continue
            wmap[cat] = min(simplex,
                            key=lambda w: wape(w[0]*cc.g + w[1]*cc.l + w[2]*cc.rf, A, sa))
        W = te.category_name.map(wmap)
        te["ens"] = (W.map(lambda x: x[0]) * te.g
                     + W.map(lambda x: x[1]) * te.l
                     + W.map(lambda x: x[2]) * te.rf).clip(lower=0)
        parts.append(te)
    S = pd.concat(parts, ignore_index=True)

    # top 10 categories by total non-excluded actual volume (lag 4, pooled)
    vol = S.groupby("category_name").actual.sum().sort_values(ascending=False)
    top10 = vol.index[:10].tolist()
    print("\nTop-10 categories (by lag-4 volume):", top10)

    # per-origin per-category result table (GTIN lag-4)
    def stats(gg):
        A = gg.actual.values; sa = A.sum()
        if sa <= 0: return None
        ao = 1 - wape(gg.ens.values, A, sa); al = 1 - wape(gg.rf.values, A, sa)
        bo = (gg.ens.values - A).sum()/sa; bl = (gg.rf.values - A).sum()/sa
        return dict(acc_o=ao, acc_l=al, bias_o=bo, bias_l=bl,
                    acc_win=ao>al, bias_win=abs(bo)<abs(bl))

    rows = []
    for snap in SN:
        for cat in top10:
            gg = S[(S.snapshot == snap) & (S.category_name == cat)]
            r = stats(gg)
            if r is None: continue
            rows.append({"snapshot": snap, "category": cat, **r})
    R = pd.DataFrame(rows)
    R["both_win"] = R.acc_win & R.bias_win
    R.to_csv(f"{OUT}/per_origin_top10.csv", index=False)

    # Matrix: rows = category, cols = snapshot, cells = WIN/L/etc
    mat = R.pivot_table(index="category", columns="snapshot",
                        values="both_win", aggfunc="first")
    mat = mat.reindex(top10)  # preserve volume order
    mat = mat.applymap(lambda v: "WIN" if v else "lose")
    mat["consistent_wins"] = (R.pivot_table(index="category", columns="snapshot",
                                            values="both_win", aggfunc="first")
                              .reindex(top10).sum(axis=1).astype(int).astype(str) + "/7")
    print("\n=== Per-origin BOTH-METRICS WIN matrix (rf-ensemble, GTIN lag-4) ===")
    pd.set_option("display.width", 240, "display.max_columns", None)
    print(mat.to_string())

    # Diagnostic: acc & |bias| gap per cell, for the 7 categories that win pooled
    pool = (R.groupby("category")
            .apply(lambda x: pd.Series({"acc_o": (x.acc_o*0).sum() or  # placeholder
                                        None})))  # we'll recompute properly below
    # proper pooled: pool actuals/forecasts across origins
    pool_rows = []
    for cat, cc in S.groupby("category_name"):
        if cat not in top10: continue
        A = cc.actual.values; sa = A.sum()
        if sa <= 0: continue
        pool_rows.append({"category": cat,
                          "acc_o_pool": 1 - wape(cc.ens.values, A, sa),
                          "acc_l_pool": 1 - wape(cc.rf.values, A, sa),
                          "bias_o_pool": (cc.ens.values - A).sum()/sa,
                          "bias_l_pool": (cc.rf.values - A).sum()/sa})
    P = pd.DataFrame(pool_rows).set_index("category").reindex(top10)
    P["pooled_win"] = (P.acc_o_pool > P.acc_l_pool) & (P.bias_o_pool.abs() < P.bias_l_pool.abs())
    P.to_csv(f"{OUT}/pooled_top10.csv")
    print("\n=== Pooled (all 7 origins) — for reference ===")
    print(P.round(3).to_string())

    print(f"\nSaved -> {OUT}/per_origin_top10.csv  +  pooled_top10.csv")


if __name__ == "__main__":
    main()
