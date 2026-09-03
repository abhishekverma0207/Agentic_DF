"""Fix the DIQ under-bias at the model level: sweep the deep model's training loss.
MAE targets the median (-> under-bias on right-skewed demand); RMSE/MSE/Huber target the
mean (like LEGO's RF). For each loss, train the global TFT (origin 202602 -> test 202603-15)
and report per-category + sales-weighted ACCURACY and BIAS vs LEGO. Goal: cut |bias| while
holding accuracy. Saves deep_TFT_<loss>_fc.parquet for the winner to feed the route."""
import sys, os; sys.path.insert(0, ".")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from utils.deep_forecast import train_global_deep
from utils import lego_eval

LOSSES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["MAE", "RMSE", "Huber"]
CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]
base = lego_eval.load_eval_base()


def load():
    fr = []
    for c in CATS:
        d = pd.read_parquet(f"sourcedata/THAILAND/by_category/{c}.parquet")
        d = d[d["key"].astype(str).str.endswith("_E1016")].copy(); d["__cat"] = c; fr.append(d)
    return pd.concat(fr, ignore_index=True)


def ev(fc, label):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    def ab(s):
        a = s["actual"].values; f = s["ours"].fillna(0).values; t = np.abs(a).sum()
        return (1 - np.abs(f - a).sum() / t, (f - a).sum() / t)
    rows = []
    for c in sorted(p.Category.unique()):
        pc = p[p.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego))
        rows.append((c, pc.actual.sum(), oa, ob, la, lb))
    d = pd.DataFrame(rows, columns=["c", "v", "oa", "ob", "la", "lb"]); w = d.v / d.v.sum()
    print(f"  {label:8s} | acc {(d.oa*w).sum():.3f} (LEGO {(d.la*w).sum():.3f}, win {int((d.oa>d.la).sum())}/9) "
          f"| bias {(d.ob*w).sum():+.3f} (LEGO {(d.lb*w).sum():+.2f}, |b|-win {int((d.ob.abs()<d.lb.abs()).sum())}/9)", flush=True)
    return d


def main():
    df = load()
    print(f"deep loss sweep on {df['key'].nunique()} E1016 series; losses={LOSSES}\n", flush=True)
    print("  loss     | sales-wtd accuracy        | sales-wtd bias", flush=True)
    print("  " + "-" * 70, flush=True)
    results = {}
    for ln in LOSSES:
        fc = train_global_deep(df, key_col="key", period_col="year_week", target_col="Actuals",
                               category_col="Category", origin_period=202602, horizon=13,
                               model_name="TFT", max_steps=2000, loss_name=ln)
        fc.to_parquet(f"artifacts_th_local/deep_TFT_{ln}_fc.parquet", index=False)
        results[ln] = ev(fc, ln)
    # per-category bias table for the winner-candidates
    print("\nper-category bias (DIQ) by loss vs LEGO:", flush=True)
    legob = results[LOSSES[0]].set_index("c")["lb"]
    hdr = "  " + f"{'category':24s}" + "".join(f"{l:>8s}" for l in LOSSES) + f"{'LEGO':>8s}"
    print(hdr, flush=True)
    for c in results[LOSSES[0]]["c"]:
        line = f"  {c:24s}" + "".join(f"{results[l].set_index('c').loc[c,'ob']:+8.2f}" for l in LOSSES) + f"{legob[c]:+8.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
