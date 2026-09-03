"""Score the clean baseline run: per category load the raw DMH forecast
(<CAT>/model_artifacts/inference_forecast.csv), re-apply the adaptive seasonal
correction, and eval raw vs corrected vs per-key-rescue vs LEGO at t45. Picks the
best LEGITIMATE per-category forecast and writes artifacts_th_local/final_forecast.parquet."""
import sys, glob; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from utils import lego_eval, seasonal_blend

CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]
CATNAME = {"FOODS": "FOODS", "FABRIC_CLEANING": "FABRIC CLEANING", "FACE": "FACE",
           "HOME_AND_HYGIENE": "HOME & HYGIENE", "DEODORANTS_AND_FRAGRANCES": "DEODORANTS & FRAGRANCES"}
cn = lambda c: CATNAME.get(c, c.replace("_", " "))
base = lego_eval.load_eval_base()


def load_cat(c):
    """Return (raw_df, corrected_df) for category c at [key, year_week, predicted]."""
    f = glob.glob(f"artifacts_th_local/final/{c}/**/inference_forecast.csv", recursive=True)
    if not f:
        return None, None
    raw = pd.read_csv(f[0])
    raw = raw[["key", "year_week", "predicted"]].copy()
    src = f"sourcedata/THAILAND/by_category/{c}.parquet"
    hist = pd.read_parquet(src, columns=["key", "year_week", "Actuals", "Category"])
    d = raw.copy(); d["Category"] = d["key"].map(hist.drop_duplicates("key").set_index("key")["Category"])
    corr = seasonal_blend.seasonal_correct(d, hist, level="adaptive")[["key", "year_week", "predicted"]]
    return raw, corr


def t45(fc, c):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted")
    pc = p[(p["Category"] == cn(c)) & (p.t.isin([4, 5]))]
    a = pc["actual"].values; f = pc["ours"].fillna(0).values; t = a.sum()
    return (1 - np.abs(f - a).sum() / t, (f - a).sum() / t, pc["actual"].sum()) if t else (np.nan, np.nan, 0)


def legoacc(c):
    p = lego_eval.attach_forecast(base, base.assign(key=base.cs + "_E1016", year_week=base.Shipment_Week, predicted=base.lego)[["key", "year_week", "predicted"]])
    pc = p[(p["Category"] == cn(c)) & (p.t.isin([4, 5]))]
    a = pc["actual"].values; f = pc["ours"].fillna(0).values; t = a.sum()
    return 1 - np.abs(f - a).sum() / t if t else np.nan


print(f"{'category':24s} {'LEGO':>6s} | {'raw':>7s} {'corr':>7s} {'perkey':>7s} | pick   verdict")
print("-" * 80)
final, rows = [], []
for c in CATS:
    raw, corr = load_cat(c)
    if raw is None:
        print(f"{cn(c):24s}  (missing)"); continue
    L = legoacc(c)
    cand = {"raw": (raw, t45(raw, c)), "corr": (corr, t45(corr, c))}
    pk = f"artifacts_th_local/{c}_hybrid/_hybrid_oos_fc.parquet"
    if glob.glob(pk):
        d = pd.read_parquet(pk)[["key", "year_week", "predicted"]]
        cand["perkey"] = (d, t45(d, c))
    # routing: raw global by default; SPLY-corrected only for reliably-seasonal categories
    # (HOME/SKIN — correction hurts erratic ones); per-key rescue for FABRIC & best-global FOODS
    PICK = {"FABRIC_CLEANING": "perkey", "FOODS": "perkey",
            "HOME_AND_HYGIENE": "corr", "SKIN_CLEANSING": "corr"}
    pick = PICK.get(c, "raw")
    if pick not in cand:
        pick = "corr" if "corr" in cand else "raw"
    fdf, (acc, bias, vol) = cand[pick]
    final.append(fdf); rows.append((c, vol, acc, L, pick))
    def s(k): return f"{cand[k][1][0]:+.3f}" if k in cand else "   -   "
    print(f"{cn(c):24s} {L:+.3f} | {s('raw')} {s('corr')} {s('perkey')} | {pick:6s} {'WIN' if acc>L else 'lose'}")

fc = pd.concat(final, ignore_index=True)
fc.to_parquet("artifacts_th_local/final_forecast.parquet", index=False)
df = pd.DataFrame(rows, columns=["c", "vol", "acc", "lego", "pick"]); w = df.vol / df.vol.sum()
print("-" * 80)
print(f"SALES-WTD t45: DIQ {(df.acc*w).sum():.3f} vs LEGO {(df.lego*w).sum():.3f} | "
      f"wins {int((df.acc>df.lego).sum())}/9 ({df.loc[df.acc>df.lego,'vol'].sum()/df.vol.sum():.0%} of sales)")
print("wrote artifacts_th_local/final_forecast.parquet")
