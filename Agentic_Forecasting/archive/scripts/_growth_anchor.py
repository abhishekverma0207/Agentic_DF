"""Test a legitimate (no test-window leakage) growth-aware level anchor to close the bias gap.
Anchor(key, w) = actual(key, w-52) * g_cat, where g_cat = recent YoY growth from PRE-origin history
(sum actuals 202541..202602 / sum actuals 202441..202502). Blend cat-select with the anchor at a
few weights and report acc/bias vs LEGO at t45 and 13wk. Uses only pre-origin actuals -> no leakage."""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
CATS = sorted(c for c in base.Category.unique() if c != "ICE CREAM")
sel = pd.read_parquet("artifacts_th_local/final_forecast_best.parquet"); sel["key"] = sel["key"].astype(str)
sel["year_week"] = sel["year_week"].astype(int)

# --- load full E1016 history for SPLY + growth ---
fr = []
for c in CATS:
    f = f"sourcedata/THAILAND/by_category/{c}.parquet"
    if not glob.glob(f):
        continue
    d = pd.read_parquet(f); d = d[d["key"].astype(str).str.endswith("_E1016")]
    d = d[["key", "year_week", "Actuals", "Category"]].copy(); fr.append(d)
hist = pd.concat(fr, ignore_index=True)
hist["key"] = hist["key"].astype(str); hist["year_week"] = pd.to_numeric(hist["year_week"], errors="coerce").astype("int64")
hist["Actuals"] = pd.to_numeric(hist["Actuals"], errors="coerce").fillna(0.0)


def yw_minus_52(yw):
    y, w = yw // 100, yw % 100
    return (y - 1) * 100 + w   # ISO-week aligned same-week-last-year (approx; fine for weekly demand)


# --- per-category YoY growth from pre-origin windows ---
def win_sum(lo, hi):
    return hist[(hist.year_week >= lo) & (hist.year_week <= hi)].groupby("Category")["Actuals"].sum()
g = (win_sum(202541, 202602) / win_sum(202441, 202502).replace(0, np.nan)).clip(0.7, 1.6)
print("per-category YoY growth g_cat:", {k: round(v, 2) for k, v in g.items()})

# --- SPLY anchor for the forecast weeks ---
fc_weeks = sorted(sel.year_week.unique())
sply_rows = []
hidx = hist.set_index(["key", "year_week"])["Actuals"]
for w in fc_weeks:
    src = yw_minus_52(w)
    s = hist[hist.year_week == src][["key", "Category", "Actuals"]].copy()
    s["year_week"] = w; s["anchor"] = s["Actuals"] * s["Category"].map(g).fillna(1.0)
    sply_rows.append(s[["key", "year_week", "anchor"]])
anchor = pd.concat(sply_rows, ignore_index=True)
m = sel.merge(anchor, on=["key", "year_week"], how="left")
m["anchor"] = m["anchor"].fillna(m["predicted"])   # no SPLY -> fall back to model


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(tot, 1e-9), (f - a).sum() / max(tot, 1e-9))


def board(fc, name):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    line = [name]
    for tag, tsel in [("t45", [4, 5]), ("13wk", list(range(1, 14)))]:
        q = p[p.t.isin(tsel)]; rows = []
        for c in CATS:
            pc = q[q.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego)); rows.append((pc.actual.sum(), oa, la, ob, lb))
        d = pd.DataFrame(rows, columns=["v", "oa", "la", "ob", "lb"]); w = d.v / d.v.sum()
        line.append("%s acc %.3f/%.3f bias %+.3f/%+.3f BOTH %d/9" % (
            tag, (d.oa * w).sum(), (d.la * w).sum(), (d.ob * w).sum(), (d.lb * w).sum(),
            int(((d.oa > d.la) & (d.ob.abs() < d.lb.abs())).sum())))
    print("  " + line[0].ljust(14) + " | ".join(line[1:]))


print("LEGO ref + blends of cat-select with growth anchor:")
board(sel.rename(columns={"predicted": "predicted"}), "cat-select")
for w in [0.2, 0.35, 0.5, 0.65]:
    b = m.copy(); b["predicted"] = (1 - w) * b["predicted"] + w * b["anchor"]
    board(b[["key", "year_week", "predicted"]], f"blend w={w}")
# pure anchor
b = m.copy(); b["predicted"] = b["anchor"]; board(b[["key", "year_week", "predicted"]], "pure anchor")
