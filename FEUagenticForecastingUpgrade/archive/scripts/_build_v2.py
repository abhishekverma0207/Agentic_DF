"""Build the v2 DIQ forecast = GBM hybrid + deep TFT combined. Compares a 50/50 blend
vs per-category model selection (both metrics), writes the chosen one to
artifacts_th_local/final_forecast.parquet. Usage: _build_v2.py [blend|route]"""
import sys, glob; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from utils import lego_eval

MODE = sys.argv[1] if len(sys.argv) > 1 else "compare"
DEEPFILE = sys.argv[2] if len(sys.argv) > 2 else "artifacts_th_local/deep_TFT_fc.parquet"
GBMFILE = sys.argv[3] if len(sys.argv) > 3 else "artifacts_th_local/final_forecast_gbm_v1.parquet"
base = lego_eval.load_eval_base()
g = pd.read_parquet(GBMFILE)[["key", "year_week", "predicted"]].rename(columns={"predicted": "g"})
d = pd.read_parquet(DEEPFILE)[["key", "year_week", "predicted"]].rename(columns={"predicted": "d"})
g["key"] = g["key"].astype(str); g["year_week"] = g["year_week"].astype(int)
d["key"] = d["key"].astype(str); d["year_week"] = d["year_week"].astype(int)
m = g.merge(d, on=["key", "year_week"], how="outer")
m["g"] = m["g"].fillna(m["d"]); m["d"] = m["d"].fillna(m["g"])  # fallback if only one has the key


def evcat(fcol, tag):
    fc = m[["key", "year_week", fcol]].rename(columns={fcol: "predicted"})
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    def acc(s, col):
        a = s["actual"].values; f = (s["ours"].fillna(0) if col == "ours" else s[col]).values
        t = np.abs(a).sum(); return 1 - np.abs(f - a).sum() / t if t else np.nan
    rows = []
    for c in sorted(p.Category.unique()):
        pc = p[p.Category == c]; p45 = pc[pc.t.isin([4, 5])]
        rows.append((c, pc.actual.sum(), acc(p45, "ours"), acc(p45, "lego"), acc(pc, "ours"), acc(pc, "lego")))
    df = pd.DataFrame(rows, columns=["c", "v", "o45", "l45", "o13", "l13"]); w = df.v / df.v.sum()
    print(f"\n[{tag}] per category (t45 ours/LEGO | 13wk ours/LEGO):")
    for _, r in df.iterrows():
        print(f"  {r.c:24s} t45 {r.o45:+.3f}/{r.l45:+.3f} {'W' if r.o45>r.l45 else 'L'} | "
              f"13wk {r.o13:+.3f}/{r.l13:+.3f} {'W' if r.o13>r.l13 else 'L'}")
    print(f"  {'SALES-WTD':24s} t45 {(df.o45*w).sum():.3f}/{(df.l45*w).sum():.3f} | 13wk {(df.o13*w).sum():.3f}/{(df.l13*w).sum():.3f}"
          f"  || t45 {int((df.o45>df.l45).sum())}/9, 13wk {int((df.o13>df.l13).sum())}/9")
    return df


# blend
m["blend"] = 0.5 * m["g"] + 0.5 * m["d"]
# per-category routing: deep TFT for the categories it wins (high-vol dense), GBM else
catmap = base[["cs", "Category"]].drop_duplicates().set_index("cs")["Category"]
m["cat"] = m["key"].str[:-6].map(catmap)
TFT_CATS = {"FOODS", "FABRIC ENHANCERS", "FACE", "HAIR CARE", "BODY", "DEODORANTS & FRAGRANCES"}
m["route"] = np.where(m["cat"].isin(TFT_CATS), m["d"], m["g"])

if MODE in ("compare", "blend"):
    evcat("blend", "BLEND 50/50")
if MODE in ("compare", "route"):
    evcat("route", "PER-CATEGORY ROUTE (TFT high-vol / GBM rest)")
evcat("g", "GBM only"); evcat("d", "TFT only")

if MODE in ("blend", "route"):
    col = "blend" if MODE == "blend" else "route"
    out = m[["key", "year_week", col]].rename(columns={col: "predicted"})
    out["predicted"] = out["predicted"].clip(lower=0)
    out.to_parquet("artifacts_th_local/final_forecast.parquet", index=False)
    print(f"\nwrote final_forecast.parquet ({MODE})")
