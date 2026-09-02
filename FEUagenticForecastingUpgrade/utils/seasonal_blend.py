"""Seasonal-level correction for TH forecasts (the proven lever vs LEGO).

The DMH under-forecasts the val->test seasonal uplift (Q1 demand is ~+13% over the
recent training tail; bias calibration learned on validation can't see it). SPLY
(same weeks last year) is the right level (test/SPLY ~ 1.04) and is KNOWN at forecast
time. So we scale each key's forecast so its forecast-period total matches its SPLY
total:

    scale = clip( sum_w actual[key, w-52] / sum_w pred[key, w],  lo, hi )

applied per-key where SPLY is reliable, else per-category (robust shrinkage fallback).
Optionally blend the corrected forecast with the SPLY/level8 anchor.

All inputs (last-year actuals) are history at the cutoff — no leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _sply_week(yw: int) -> int:
    """YYYYWW -> same week previous year (YYYYWW)."""
    yw = int(yw)
    return (yw // 100 - 1) * 100 + (yw % 100)


def seasonal_correct(
    fc: pd.DataFrame,
    hist: pd.DataFrame,
    *,
    key_col: str = "key",
    date_col: str = "year_week",
    target_col: str = "Actuals",
    pred_col: str = "predicted",
    cat_col: str = "Category",
    level: str = "adaptive",     # 'adaptive' (reliability-gated), 'key', or 'category'
    lo: float = 0.5,
    hi: float = 2.5,
    min_sply: float = 5.0,       # min SPLY volume for a trusted per-key scale ('key' mode)
    out_col: str | None = None,
) -> pd.DataFrame:
    """Return fc with `out_col` (default: overwrite pred_col) = seasonally scaled forecast."""
    out_col = out_col or pred_col
    fc = fc.copy()
    fc["_yw"] = pd.to_numeric(fc[date_col], errors="coerce").astype("Int64")
    fcw = sorted(int(w) for w in fc["_yw"].dropna().unique())
    sply_weeks = [_sply_week(w) for w in fcw]

    h = hist[[key_col, date_col, target_col] + ([cat_col] if cat_col in hist.columns else [])].copy()
    h["_yw"] = pd.to_numeric(h[date_col], errors="coerce").astype("Int64")
    hs = h[h["_yw"].isin(sply_weeks)].copy()
    hs["_v"] = pd.to_numeric(hs[target_col], errors="coerce").fillna(0.0)
    sply_tot = hs.groupby(key_col)["_v"].sum().rename("sply_tot")
    sply_pos = hs.assign(_p=hs["_v"] > 0).groupby(key_col)["_p"].sum().rename("sply_pos")
    ours_tot = fc.groupby(key_col)[pred_col].sum().rename("ours_tot")
    n_fw = max(len(fcw), 1)

    df = pd.concat([sply_tot, sply_pos, ours_tot], axis=1).fillna(0.0)
    if cat_col in fc.columns:
        df[cat_col] = df.index.map(fc[[key_col, cat_col]].drop_duplicates()
                                   .set_index(key_col)[cat_col])
    else:
        df[cat_col] = "__ALL__"
    cat_num = df.groupby(cat_col)["sply_tot"].sum()
    cat_den = df.groupby(cat_col)["ours_tot"].sum().clip(lower=1e-9)
    catscale = (cat_num / cat_den).clip(lo, hi)
    key_ratio = (df["sply_tot"] / df["ours_tot"].clip(lower=1e-9)).clip(lo, hi)
    # per-key SPLY reliability = fraction of forecast weeks with a positive SPLY value
    df["reliab"] = (df["sply_pos"] / n_fw).clip(0.0, 1.0)

    if level == "adaptive":
        # Shrink each key's scale toward 1.0 (= raw, no correction) by its SPLY
        # reliability. Stable high-volume keys (full SPLY history) get the full
        # seasonal level-lock; sparse/NPD keys (unreliable SPLY) stay ~raw — which
        # is where the pooled model already beats LEGO.
        df["scale"] = (1.0 + df["reliab"] * (key_ratio - 1.0)).clip(lo, hi)
    elif level == "key":
        trusted = (df["sply_tot"] >= min_sply) & (df["ours_tot"] > 0)
        df["scale"] = np.where(trusted, key_ratio, df[cat_col].map(catscale))
    else:  # 'category'
        df["scale"] = df[cat_col].map(catscale)
    df["scale"] = df["scale"].fillna(1.0)

    fc[out_col] = fc[pred_col] * fc[key_col].map(df["scale"]).fillna(1.0).values
    return fc.drop(columns=["_yw"])


def apply_to_forecast_file(forecast_path: str, hist_path: str, **kw) -> pd.DataFrame:
    fc = (pd.read_parquet(forecast_path) if forecast_path.endswith(".parquet")
          else pd.read_csv(forecast_path))
    hist = (pd.read_parquet(hist_path) if hist_path.endswith(".parquet")
            else pd.read_csv(hist_path))
    return seasonal_correct(fc, hist, **kw)
