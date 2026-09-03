"""Test the available levers at the 13-WEEK objective for the 3 hard top-5 categories
(FABRIC CLEANING, HAIR CARE, FOODS). Lever = per-key SPLY-seasonal overlay (already trained,
_seasonal_perkey_fc.parquet, all 13 horizons) blended with the Step-A base, plus base bias-blends
for FOODS. Reports per-category 13wk acc/bias vs LEGO for each config; picks the best per category."""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
A = pd.read_parquet("artifacts_th_local/final_forecast_catselect13.parquet"); A["key"] = A["key"].astype(str); A["year_week"] = A["year_week"].astype(int)
seas = pd.read_parquet("artifacts_th_local/_seasonal_perkey_fc.parquet"); seas["key"] = seas["key"].astype(str); seas["year_week"] = seas["year_week"].astype(int)
seas["cat"] = seas["key"].str[:-6].map(cs2cat)
RAW = {}
for f in sorted(glob.glob("artifacts_th_local/deep_TFT_*_fc.parquet")):
    nm = f.split("deep_TFT_")[1].split("_fc")[0]
    if nm != "allkeys":
        d = pd.read_parquet(f); d["key"] = d["key"].astype(str); RAW[nm] = d[["key", "year_week", "predicted"]]


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(tot, 1e-9), (f - a).sum() / max(tot, 1e-9))


def cat13(fc, c):
    cskeys = set(base.loc[base.Category == c, "cs"].astype(str) + "_E1016")
    p = lego_eval.attach_forecast(base, fc[fc.key.isin(cskeys)], pred_col="predicted"); p = p[p.Category == c]
    a, b = ab(p); la, lb = ab(p.assign(ours=p.lego))
    return a, b, la, lb


def seasonal_overlay(cat, w):
    """blend base with per-key seasonal for cat's hv keys (all 13 horizons) at weight w on seasonal."""
    s = seas[seas.cat == cat][["key", "year_week", "predicted"]].rename(columns={"predicted": "sp"})
    m = A.merge(s, on=["key", "year_week"], how="left")
    m["predicted"] = np.where(m["sp"].notna(), (1 - w) * m["predicted"] + w * m["sp"], m["predicted"])
    return m[["key", "year_week", "predicted"]]


def base_blend(cat, loss_a, loss_b, w):
    """replace cat's keys in base with w*loss_a + (1-w)*loss_b."""
    cskeys = set(base.loc[base.Category == cat, "cs"].astype(str) + "_E1016")
    bl = RAW[loss_a].merge(RAW[loss_b], on=["key", "year_week"], suffixes=("_a", "_b"))
    bl["predicted"] = w * bl["predicted_a"] + (1 - w) * bl["predicted_b"]
    bl = bl[bl.key.isin(cskeys)][["key", "year_week", "predicted"]]
    return pd.concat([bl, A[~A.key.isin(cskeys)]], ignore_index=True)


print("Format: DIQ acc/LEGO  DIQ bias/LEGO   verdict   (target: acc>LEGO AND |bias|<|LEGO|)")
for cat in ["FABRIC CLEANING", "HAIR CARE"]:
    a, b, la, lb = cat13(A, cat)
    print(f"\n=== {cat} (Step-A base: acc {a:+.3f}/{la:+.3f} bias {b:+.3f}/{lb:+.3f}) ===")
    for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
        fc = seasonal_overlay(cat, w); a, b, la, lb = cat13(fc, cat)
        v = "WIN both" if (a > la and abs(b) < abs(lb)) else ("acc" if a > la else ("bias" if abs(b) < abs(lb) else "-"))
        print(f"  seasonal w={w:.1f}: acc {a:+.3f}/{la:+.3f}  bias {b:+.3f}/{lb:+.3f}  {v}")

print("\n=== FOODS (acc appears capped ~0.575 < LEGO 0.576; scan blends) ===")
for la_, lb_ in [("Huber8", "RMSE"), ("Huber3", "RMSE"), ("Huber8", "Huber3")]:
    for w in [1.0, 0.8, 0.6, 0.4]:
        fc = base_blend("FOODS", la_, lb_, w); a, b, la, lb = cat13(fc, "FOODS")
        v = "WIN both" if (a > la and abs(b) < abs(lb)) else ("acc" if a > la else ("bias" if abs(b) < abs(lb) else "-"))
        print(f"  {la_}*{w:.1f}+{lb_}*{1-w:.1f}: acc {a:+.3f}/{la:+.3f}  bias {b:+.3f}/{lb:+.3f}  {v}")
