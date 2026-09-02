"""Why do we WIN accuracy but LOSE bias? Bias = signed sum, so a few keys we systematically
under-forecast (while LEGO captures their level) can dominate it. Decompose the DIQ-vs-LEGO bias
GAP per key: gap_contrib(key) = (DIQ_total - LEGO_total) / category_actual_total. Negative = DIQ
forecasts lower than LEGO on that key -> drives our bias loss. Rank, and check concentration."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
fc = pd.read_parquet("artifacts_th_local/final_forecast_catselect13.parquet"); fc["key"] = fc["key"].astype(str)
p = lego_eval.attach_forecast(base, fc, pred_col="predicted")

for cat in ["FABRIC CLEANING", "HAIR CARE", "FOODS"]:
    pc = p[p.Category == cat].copy()
    g = pc.groupby("cs").agg(actual=("actual", "sum"), diq=("ours", "sum"), lego=("lego", "sum")).reset_index()
    catA = g["actual"].sum()
    diq_bias = (g["diq"] - g["actual"]).sum() / catA
    lego_bias = (g["lego"] - g["actual"]).sum() / catA
    g["diq_c"] = (g["diq"] - g["actual"]) / catA          # this key's contribution to DIQ bias
    g["lego_c"] = (g["lego"] - g["actual"]) / catA
    g["gap"] = g["diq_c"] - g["lego_c"]                    # negative => DIQ under vs LEGO => drives bias loss
    g["volpct"] = g["actual"] / catA
    gap_total = diq_bias - lego_bias
    print(f"\n========== {cat}: DIQ bias {diq_bias:+.3f} vs LEGO {lego_bias:+.3f}  (gap {gap_total:+.3f}) ==========")
    drivers = g.sort_values("gap").head(12)
    print(f"  {'CS_BARCODE':16s} {'vol%':>5s} {'actual':>9s} {'DIQ':>9s} {'LEGO':>9s} {'DIQ%':>6s} {'LEGO%':>6s} {'gap%':>6s}")
    for _, r in drivers.iterrows():
        print(f"  {r.cs:16s} {r.volpct:4.1%} {r.actual:9.0f} {r.diq:9.0f} {r.lego:9.0f} "
              f"{r.diq/max(r.actual,1)-1:+5.0%} {r.lego/max(r.actual,1)-1:+5.0%} {r.gap:+6.1%}")
    # concentration: how many keys explain the gap?
    neg = g[g.gap < 0].sort_values("gap")
    cum = neg["gap"].cumsum()
    n50 = int((cum > 0.5 * gap_total).sum()) if gap_total < 0 else 0   # keys to reach 50% of the (negative) gap
    topvol = drivers["volpct"].sum()
    print(f"  -> {len(neg)} keys hurt bias vs LEGO; the worst 12 carry {drivers['gap'].sum()/gap_total:.0%} of the gap "
          f"({topvol:.0%} of category volume). bias gap is {'CONCENTRATED' if drivers['gap'].sum()/gap_total>0.6 else 'broad'}.")
    # direction check: among these drivers, is DIQ winning accuracy but losing bias?
    d = drivers.assign(diqW=lambda x: np.abs(x.diq - x.actual), legoW=lambda x: np.abs(x.lego - x.actual))
    print(f"     on these 12: DIQ abs-error sum {d.diqW.sum():.0f} vs LEGO {d.legoW.sum():.0f} "
          f"({'DIQ more accurate' if d.diqW.sum() < d.legoW.sum() else 'LEGO more accurate'} on them too)")
