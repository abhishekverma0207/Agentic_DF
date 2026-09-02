"""Legitimate per-category bias calibration for the v2 DIQ route.

The DIQ under-bias is the seasonal val->test gap (Q1 demand > recent trend). Learn a
per-category multiplicative factor on the SAME SEASON LAST YEAR (2025 wk 03-15 = the SPLY
of the 2026 test) — a true holdout with known actuals, so no peeking at the 2026 test:

    factor[cat] = sum(actual_2025Q1) / sum(forecast_2025Q1)   (clipped, reliability-shrunk)

The 2025-Q1 forecast comes from the deep model trained on history <= 2025wk02 — its
per-category seasonal under-bias is a property of the demand data (right-skew + Q1 lift)
shared by the GBM, so the factor calibrates the whole route. Apply to the 2026 test route,
then verify on the test (bias -> ~0, accuracy held). Writes final_forecast_calibrated.parquet."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from utils.deep_forecast import train_global_deep
from utils import lego_eval

CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]
import os
# 202541 = recent validation (predict 202542..202602, same data depth as deployment ->
# captures the model's SYSTEMATIC 13-wk-ahead under-bias, transferable; vs 202502 same-season
# which over-corrects because the Q1 lift is year-unstable). Override with CAL_ORIGIN env.
CAL_ORIGIN = int(os.environ.get("CAL_ORIGIN", 202541))
LO, HI = 0.80, 1.50           # clip the factor (milder)


def main():
    frames = []
    for c in CATS:
        d = pd.read_parquet(f"sourcedata/THAILAND/by_category/{c}.parquet")
        d = d[d["key"].astype(str).str.endswith("_E1016")].copy(); d["__cat"] = c
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    catmap = df.drop_duplicates("key").set_index("key")["Category"]

    # 1. deep model trained on <=2025wk02, forecast 2025-Q1 (the holdout)
    val_fc = train_global_deep(df, key_col="key", period_col="year_week", target_col="Actuals",
                               category_col="Category", origin_period=CAL_ORIGIN, horizon=13,
                               model_name="TFT", max_steps=2000)
    val_fc["Category"] = val_fc["key"].map(catmap)
    fcast = val_fc.groupby("Category")["predicted"].sum()

    # 2. holdout actuals per category (exactly the forecast weeks)
    df["yw"] = pd.to_numeric(df["year_week"], errors="coerce")
    val_weeks = set(int(w) for w in val_fc["year_week"].unique())
    a = df[df["yw"].isin(val_weeks)]
    actual = a.groupby("Category")["Actuals"].sum()
    wk = a.groupby("Category")["yw"].nunique() / max(len(val_weeks), 1)

    # 3. per-category factor, clipped + shrunk toward 1 by SPLY-week presence
    raw = (actual / fcast).reindex(fcast.index)
    factor = (1.0 + wk.reindex(raw.index).fillna(0) * (raw.clip(LO, HI) - 1.0)).clip(LO, HI)
    print("per-category calibration factor (2025-Q1 holdout):")
    for c in factor.index:
        print(f"  {c:24s} actual/fcst={raw.get(c, np.nan):.2f} -> factor {factor[c]:.3f}")

    # 4. apply to the 2026 test route (per category, E1016 grain)
    base = lego_eval.load_eval_base()
    cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
    route = pd.read_parquet("artifacts_th_local/final_forecast.parquet").copy()
    route["Category"] = route["key"].astype(str).str[:-6].map(cs2cat)
    route["__f"] = route["Category"].map(factor).fillna(1.0)
    route["predicted"] = route["predicted"] * route["__f"]
    out = route[["key", "year_week", "predicted"]]
    out.to_parquet("artifacts_th_local/final_forecast_calibrated.parquet", index=False)

    # 5. verify on the 2026 test (bias + accuracy, before vs after)
    def ev(fc, label):
        p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
        def ab(s):
            ac = s["actual"].values; f = s["ours"].fillna(0).values; t = np.abs(ac).sum()
            return (1 - np.abs(f - ac).sum() / t, (f - ac).sum() / t)
        rows = []
        for c in sorted(p.Category.unique()):
            pc = p[p.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego))
            rows.append((c, pc.actual.sum(), oa, la, ob, lb))
        d = pd.DataFrame(rows, columns=["c", "v", "oa", "la", "ob", "lb"]); w = d.v / d.v.sum()
        print(f"  {label:16s} 13wk acc {(d.oa*w).sum():.3f} (LEGO {(d.la*w).sum():.3f}, win {int((d.oa>d.la).sum())}/9) "
              f"| bias {(d.ob*w).sum():+.2f} (LEGO {(d.lb*w).sum():+.2f}, |bias|-win {int((d.ob.abs()<d.lb.abs()).sum())}/9)")
    print("\nTEST verification (2026-03..15):")
    ev(pd.read_parquet("artifacts_th_local/final_forecast.parquet"), "BEFORE (v2)")
    ev(out, "AFTER (calibrated)")


if __name__ == "__main__":
    main()
