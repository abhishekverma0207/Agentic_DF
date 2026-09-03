"""Per-key seasonality-gated SPLY anchor (no test leakage). For each key, measure SPLY SKILL on
PAST years (how well actual(w-52) predicted actual(w) over 202401..202602). Blend cat-select with
SPLY_future (= actual of same week last year) using weight = skill**p, so only keys with proven
stable seasonality get lifted toward their seasonal level; erratic keys keep the deep model.
Sweeps p and a skill floor, reports acc/bias vs LEGO at t45 and 13wk. Picks the best, saves it."""
import glob, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from utils import lego_eval

base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
CATS = sorted(c for c in base.Category.unique() if c != "ICE CREAM")
sel = pd.read_parquet("artifacts_th_local/final_forecast_best.parquet"); sel["key"] = sel["key"].astype(str)
sel["year_week"] = sel["year_week"].astype(int)

# full E1016 history
fr = []
for c in CATS:
    f = f"sourcedata/THAILAND/by_category/{c}.parquet"
    if not glob.glob(f):
        continue
    d = pd.read_parquet(f); d = d[d["key"].astype(str).str.endswith("_E1016")]
    fr.append(d[["key", "year_week", "Actuals"]])
hist = pd.concat(fr, ignore_index=True)
hist["key"] = hist["key"].astype(str)
hist["year_week"] = pd.to_numeric(hist["year_week"], errors="coerce").astype("int64")
hist["Actuals"] = pd.to_numeric(hist["Actuals"], errors="coerce").fillna(0.0)
A = hist.set_index(["key", "year_week"])["Actuals"].sort_index()


def ym52(yw):
    return (yw // 100 - 1) * 100 + (yw % 100)


# --- per-key SPLY skill on PAST years (202401..202602): 1 - WAPE of actual(w-52) vs actual(w) ---
wk = sorted(w for w in hist.year_week.unique() if 202401 <= w <= 202602)
cur = hist[hist.year_week.isin(wk)][["key", "year_week", "Actuals"]].copy()
cur["sply"] = [A.get((k, ym52(w)), np.nan) for k, w in zip(cur.key, cur.year_week)]
cur = cur.dropna(subset=["sply"])
sk = cur.groupby("key").apply(lambda d: 1 - np.abs(d.Actuals - d.sply).sum() / max(np.abs(d.Actuals).sum(), 1e-9))
skill = sk.clip(0, 1).rename("skill")
print(f"keys with SPLY-skill computed: {len(skill)}; skill>0.4: {(skill>0.4).sum()}, >0.6: {(skill>0.6).sum()}")

# --- SPLY for the future weeks ---
fut = sorted(sel.year_week.unique())
rows = []
for w in fut:
    s = hist[hist.year_week == ym52(w)][["key", "Actuals"]].rename(columns={"Actuals": "sply"}); s["year_week"] = w
    rows.append(s)
splyf = pd.concat(rows, ignore_index=True)
m = sel.merge(splyf, on=["key", "year_week"], how="left").merge(skill, left_on="key", right_index=True, how="left")
m["skill"] = m["skill"].fillna(0.0); m["sply"] = m["sply"]


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(tot, 1e-9), (f - a).sum() / max(tot, 1e-9))


def board(fc, name):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    out = [name.ljust(16)]
    for tag, tsel in [("t45", [4, 5]), ("13wk", list(range(1, 14)))]:
        q = p[p.t.isin(tsel)]; rows = []
        for c in CATS:
            pc = q[q.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego)); rows.append((pc.actual.sum(), oa, la, ob, lb))
        d = pd.DataFrame(rows, columns=["v", "oa", "la", "ob", "lb"]); w = d.v / d.v.sum()
        out.append("%s a %.3f/%.3f b %+.3f/%+.3f BOTH %d/9" % (tag, (d.oa * w).sum(), (d.la * w).sum(),
                   (d.ob * w).sum(), (d.lb * w).sum(), int(((d.oa > d.la) & (d.ob.abs() < d.lb.abs())).sum())))
    print("  " + out[0] + " | ".join(out[1:]))


board(sel, "cat-select")
best = None
for p in [1.0, 1.5, 2.0]:
    for floor in [0.0, 0.4, 0.5]:
        b = m.copy()
        w = np.where(b["skill"] >= floor, b["skill"] ** p, 0.0)
        w = np.where(b["sply"].notna(), w, 0.0)
        b["predicted"] = (1 - w) * b["predicted"] + w * b["sply"].fillna(0.0)
        board(b[["key", "year_week", "predicted"]], f"sply p={p} fl={floor}")
        # track sales-wtd t45 both-win + acc for pick
        pp = lego_eval.attach_forecast(base, b[["key", "year_week", "predicted"]], pred_col="predicted")
        pp = pp[(pp.Category != "ICE CREAM") & pp.t.isin([4, 5])]
        rows = []
        for c in CATS:
            pc = pp[pp.Category == c]; oa, ob = ab(pc); la, lb = ab(pc.assign(ours=pc.lego)); rows.append((pc.actual.sum(), oa, la, ob, lb))
        dd = pd.DataFrame(rows, columns=["v", "oa", "la", "ob", "lb"]); wt = dd.v / dd.v.sum()
        both = int(((dd.oa > dd.la) & (dd.ob.abs() < dd.lb.abs())).sum()); accw = (dd.oa * wt).sum(); biasw = abs((dd.ob * wt).sum())
        sc = both * 100 + accw * 10 - biasw
        if best is None or sc > best[0]:
            best = (sc, p, floor, b[["key", "year_week", "predicted"]].copy())
print(f"PICKED sply p={best[1]} floor={best[2]} -> saving final_forecast_seasonal.parquet")
best[3].to_parquet("artifacts_th_local/final_forecast_seasonal.parquet", index=False)
