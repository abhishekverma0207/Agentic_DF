"""Last cheap lever: diverse ensembles for the 3 hard top-5 categories at 13wk. Averaging across
models with uncorrelated errors usually buys accuracy for free. Tries mean-ensembles of the deep
loss variants + D3 + per-key seasonal, reports 13wk acc/bias vs LEGO. Picks any that beats both."""
import glob, itertools, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
cs2cat = base.drop_duplicates("cs").set_index("cs")["Category"]
M = {}
for f in sorted(glob.glob("artifacts_th_local/deep_TFT_*_fc.parquet")):
    nm = f.split("deep_TFT_")[1].split("_fc")[0]
    if nm != "allkeys":
        d = pd.read_parquet(f); d["key"] = d["key"].astype(str); M[nm] = d[["key", "year_week", "predicted"]]
seas = pd.read_parquet("artifacts_th_local/_seasonal_perkey_fc.parquet"); seas["key"] = seas["key"].astype(str)
seas["year_week"] = seas["year_week"].astype(int); seas["cat"] = seas["key"].str[:-6].map(cs2cat)


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(tot, 1e-9), (f - a).sum() / max(tot, 1e-9))


def cat13(fc, c):
    cskeys = set(base.loc[base.Category == c, "cs"].astype(str) + "_E1016")
    p = lego_eval.attach_forecast(base, fc[fc.key.isin(cskeys)], pred_col="predicted"); p = p[p.Category == c]
    a, b = ab(p); la, lb = ab(p.assign(ours=p.lego)); return a, b, la, lb


def mean_ens(names, cat, seasonal_w=0.0):
    cskeys = set(base.loc[base.Category == cat, "cs"].astype(str) + "_E1016")
    frames = [M[n][M[n].key.isin(cskeys)].set_index(["key", "year_week"])["predicted"] for n in names]
    ens = pd.concat(frames, axis=1).mean(axis=1).rename("predicted").reset_index()
    if seasonal_w > 0:
        s = seas[seas.cat == cat][["key", "year_week", "predicted"]].rename(columns={"predicted": "sp"})
        ens = ens.merge(s, on=["key", "year_week"], how="left")
        ens["predicted"] = np.where(ens["sp"].notna(), (1 - seasonal_w) * ens["predicted"] + seasonal_w * ens["sp"], ens["predicted"])
        ens = ens[["key", "year_week", "predicted"]]
    return ens


SUBSETS = [["MAE", "Huber8", "RMSE"], ["Huber", "Huber8", "RMSE"], ["MAE", "Huber", "Huber8", "Huber3", "RMSE"],
           ["Huber8", "RMSE", "d3"], ["MAE", "Huber8", "RMSE", "d3"], ["Huber8", "Huber3"], ["MAE", "Huber8"]]
for cat in ["FABRIC CLEANING", "HAIR CARE", "FOODS"]:
    print(f"\n=== {cat} ensembles (13wk; target acc>LEGO AND |bias|<|LEGO|) ===")
    for names in SUBSETS:
        if not all(n in M for n in names):
            continue
        for sw in [0.0, 0.3]:
            fc = mean_ens(names, cat, sw); a, b, la, lb = cat13(fc, cat)
            v = "WIN both" if (a > la and abs(b) < abs(lb)) else ("acc" if a > la else ("bias" if abs(b) < abs(lb) else "-"))
            sfx = f"+seas{sw}" if sw else ""
            print(f"  mean({'+'.join(names)}){sfx}: acc {a:+.3f}/{la:+.3f}  bias {b:+.3f}/{lb:+.3f}  {v}")
