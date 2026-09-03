"""Deep multi-horizon forecaster (the balanced-across-13-weeks model). A GLOBAL neural
model over all E1016 series forecasts all 13 horizons JOINTLY (vs our 13 independent
per-horizon GBMs), with known-future inputs (calendar/holiday/promo) + static category
covariates -> horizon-consistency. Trains N-HiTS and/or TFT on MPS, scores vs LEGO at
both t+4&5 and 13-week. Usage: _deep_forecast.py [NHITS|TFT|TIDE] [max_steps]"""
import os, sys, glob, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TFT, TiDE
from neuralforecast.losses.pytorch import MAE, HuberMQLoss
from utils import lego_eval

MODEL = sys.argv[1] if len(sys.argv) > 1 else "NHITS"
MAXSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
H, INPUT = 13, 52
CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]
ORIGIN = 202602  # last history week; forecast 202603..202615
PAT = ["songkran", "bucha", "festive", "holiday", "promo", "discount", "season",
       "priceoff", "off_invoice", "stringency", "summer", "monsoon", "winter"]


def yw_to_ds(yw):
    return pd.to_datetime(yw.astype(int).astype(str) + "1", format="%G%V%u")


def ds_to_yw(ds):
    iso = ds.dt.isocalendar()
    return (iso["year"] * 100 + iso["week"]).astype(int)


def main():
    t0 = time.time()
    allkeys = os.environ.get("DEEP_ALLKEYS") == "1"  # train on ALL accounts (4x data), still eval E1016
    # --- load panel across categories ---
    frames = []
    for c in CATS:
        d = pd.read_parquet(f"sourcedata/THAILAND/by_category/{c}.parquet")
        if not allkeys:
            d = d[d["key"].astype(str).str.endswith("_E1016")].copy()
        else:
            d = d.copy()
        d["__cat"] = c
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["year_week"] = pd.to_numeric(df["year_week"], errors="coerce")
    df = df[df["year_week"].notna() & (df["year_week"] <= 202615)].copy()
    df["unique_id"] = df["key"].astype(str)
    df["ds"] = yw_to_ds(df["year_week"])
    df["y"] = pd.to_numeric(df["Actuals"], errors="coerce").fillna(0).clip(lower=0)
    # --- known-future covariates: calendar + curated promo/holiday (NO sales-derived) ---
    woy = df["ds"].dt.isocalendar().week.astype(int)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    df["is_q1"] = (woy <= 15).astype(float)
    cov = [c for c in df.columns if any(p in c.lower() for p in PAT)
           and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    # keep covariates with signal (not all-constant) and present in the future weeks
    fut_mask = df["year_week"] > ORIGIN
    cov = [c for c in cov if df.loc[fut_mask, c].notna().any() and df[c].std() > 1e-6][:14]
    futr = ["woy_sin", "woy_cos", "is_q1"] + cov
    for c in futr:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # aggregate to ONE row per series x week (sum y, first covariate value)
    agg = {"y": "sum", "year_week": "first", **{c: "first" for c in futr}}
    g = df.groupby(["unique_id", "ds"], as_index=False).agg(agg)
    # --- static: category one-hot + log mean volume ---
    cat_oh = pd.get_dummies(df.groupby("unique_id")["__cat"].first()).astype(float)
    vol = np.log1p(g[g["year_week"] <= ORIGIN].groupby("unique_id")["y"].mean())
    static = cat_oh.copy(); static["logvol"] = vol.reindex(static.index).fillna(0)
    static.index.name = "unique_id"; static = static.reset_index()
    stat_cols = [c for c in static.columns if c != "unique_id"]
    train = g[g["year_week"] <= ORIGIN][["unique_id", "ds", "y"] + futr]
    futr_df = g[g["year_week"] > ORIGIN][["unique_id", "ds"] + futr]
    # require each series to have enough history + a full future block
    good = futr_df.groupby("unique_id").size()
    good = good[good == H].index
    hl = train.groupby("unique_id").size()
    good = good.intersection(hl[hl >= 40].index)  # need >=40 wks history (52 window + 13 val w/ padding)
    train = train[train["unique_id"].isin(good)]
    futr_df = futr_df[futr_df["unique_id"].isin(good)]
    static = static[static["unique_id"].isin(good)]
    print(f"[deep] {MODEL}: {train['unique_id'].nunique()} series, {len(futr)} futr covs, "
          f"{len(stat_cols)} static; prep {time.time()-t0:.0f}s", flush=True)

    # --- model ---
    common = dict(h=H, input_size=INPUT, futr_exog_list=futr, stat_exog_list=stat_cols,
                  scaler_type="robust", loss=MAE(), max_steps=MAXSTEPS, val_check_steps=50,
                  early_stop_patience_steps=4, enable_progress_bar=False, logger=False,
                  start_padding_enabled=True, accelerator="mps", devices=1, batch_size=64)
    if MODEL == "NHITS":
        model = NHITS(**common, n_pool_kernel_size=[2, 2, 1], n_freq_downsample=[4, 2, 1])
    elif MODEL == "TiDE":
        model = TiDE(**common)
    else:
        common.update(batch_size=32)
        model = TFT(**common, hidden_size=96, n_head=4)
    nf = NeuralForecast(models=[model], freq="W-MON")
    nf.fit(df=train, static_df=static, val_size=H)
    fc = nf.predict(futr_df=futr_df)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    fc["year_week"] = ds_to_yw(fc["ds"])
    fc = fc.rename(columns={MODEL: "predicted", "unique_id": "key"})[["key", "year_week", "predicted"]]
    fc["predicted"] = fc["predicted"].clip(lower=0)
    out = f"artifacts_th_local/deep_{MODEL}{'_allkeys' if allkeys else ''}_fc.parquet"; fc.to_parquet(out, index=False)

    # --- eval vs LEGO: t45 + 13wk per category ---
    base = lego_eval.load_eval_base()
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    def acc(s, col):
        a = s["actual"].values; f = (s["ours"].fillna(0) if col == "ours" else s[col]).values
        t = np.abs(a).sum(); return 1 - np.abs(f - a).sum() / t if t else np.nan
    print(f"\n[{MODEL}] vs LEGO ({time.time()-t0:.0f}s, {out}):", flush=True)
    print(f"  {'category':24s} | t45: deep  LEGO | 13wk: deep  LEGO", flush=True)
    rows = []
    for c in sorted(p.Category.unique()):
        pc = p[p.Category == c]; p45 = pc[pc.t.isin([4, 5])]
        o45, l45 = acc(p45, "ours"), acc(p45, "lego"); o13, l13 = acc(pc, "ours"), acc(pc, "lego")
        rows.append((c, pc.actual.sum(), o45, l45, o13, l13))
        print(f"  {c:24s} | {o45:+.3f} {l45:+.3f} {'W' if o45>l45 else 'L'} | {o13:+.3f} {l13:+.3f} {'W' if o13>l13 else 'L'}", flush=True)
    d = pd.DataFrame(rows, columns=["c", "v", "o45", "l45", "o13", "l13"]); w = d.v / d.v.sum()
    print(f"  {'SALES-WTD':24s} | t45 {(d.o45*w).sum():+.3f} {(d.l45*w).sum():+.3f}   | "
          f"13wk {(d.o13*w).sum():+.3f} {(d.l13*w).sum():+.3f}", flush=True)
    print(f"  WINS: t45 {int((d.o45>d.l45).sum())}/9, 13wk {int((d.o13>d.l13).sum())}/9", flush=True)


if __name__ == "__main__":
    main()
