"""Final TH scorecard: load each category's hybrid forecast (_hybrid_oos_fc.parquet),
eval vs LEGO per category x {t45, 13wk} for 1-WAPE & bias, plus the sales-weighted
overall (weighted by E1016 actual volume, t45 the priority). Prints WIN/LOSS verdicts."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from utils import lego_eval

CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]
CATNAME = {"FOODS": "FOODS", "FABRIC_CLEANING": "FABRIC CLEANING", "FACE": "FACE",
           "HOME_AND_HYGIENE": "HOME & HYGIENE", "DEODORANTS_AND_FRAGRANCES": "DEODORANTS & FRAGRANCES"}


def cname(c):
    return CATNAME.get(c, c.replace("_", " "))


def main():
    base = lego_eval.load_eval_base()
    frames = []
    for c in CATS:
        fp = f"artifacts_th_local/{c}_hybrid/_hybrid_oos_fc.parquet"
        if glob.glob(fp):
            d = pd.read_parquet(fp)
            d["__cat"] = cname(c)
            frames.append(d[["key", "year_week", "predicted", "__cat"]])
        else:
            print(f"  [missing] {c}", flush=True)
    allfc = pd.concat(frames, ignore_index=True)
    p = lego_eval.attach_forecast(base, allfc, pred_col="predicted")

    def ab(s):
        a = s["actual"].values; f = s["ours"].fillna(0).values; t = a.sum()
        return (1 - np.abs(f - a).sum() / t if t else np.nan, (f - a).sum() / t if t else np.nan)

    rows = []
    print(f"\n{'category':24s} | t45 acc  bias  vs LEGO  | 13wk acc bias  vs LEGO | t45 verdict", flush=True)
    print("-" * 100, flush=True)
    for c in CATS:
        cn = cname(c)
        pc = p[p["Category"] == cn]
        if pc.empty:
            continue
        p45 = pc[pc.t.isin([4, 5])]
        a45, b45 = ab(p45); l45, lb45 = ab(p45.assign(ours=p45["lego"]))
        a13, b13 = ab(pc); l13, lb13 = ab(pc.assign(ours=pc["lego"]))
        vol = pc["actual"].sum()
        win45 = (a45 > l45) and (abs(b45) < abs(lb45))
        wina = a45 > l45
        verdict = "WIN both" if win45 else ("WIN acc" if wina else "lose")
        rows.append(dict(cat=cn, vol=vol, a45=a45, b45=b45, l45=l45, a13=a13, l13=l13, wina=wina, win45=win45))
        print(f"{cn:24s} | {a45:.3f} {b45:+.2f} (L {l45:.3f}) | {a13:.3f} {b13:+.2f} (L {l13:.3f}) | {verdict}", flush=True)

    df = pd.DataFrame(rows)
    # sales-weighted overall (t45 priority)
    w = df["vol"] / df["vol"].sum()
    o45 = (df["a45"] * w).sum(); ol45 = (df["l45"] * w).sum()
    o13 = (df["a13"] * w).sum(); ol13 = (df["l13"] * w).sum()
    print("-" * 100, flush=True)
    print(f"{'SALES-WEIGHTED (t45 prio)':24s} | OURS t45 {o45:.3f} vs LEGO {ol45:.3f} "
          f"({'WIN' if o45 > ol45 else 'lose'}) | 13wk {o13:.3f} vs {ol13:.3f}", flush=True)
    nwin = int(df["wina"].sum()); nboth = int(df["win45"].sum())
    wonvol = df.loc[df["wina"], "vol"].sum() / df["vol"].sum()
    print(f"\n  t45 accuracy WINS: {nwin}/{len(df)} categories ({wonvol:.0%} of E1016 sales); "
          f"WIN both axes: {nboth}/{len(df)}", flush=True)
    df.to_parquet("notebooks/rca_outputs/_final_scorecard.parquet", index=False)


if __name__ == "__main__":
    main()
