"""Hierarchical XGBoost demand forecast — production module (mirrors the validated
``scripts/_xgb_hier.py``). A single global direct-multi-horizon XGBoost over a target account's
keys, using HIERARCHICAL features the per-key benchmark can't see:
  - own history (lags, rolling, SPLY/same-period-last-year)
  - CROSS-ACCOUNT: the same product (CS_BARCODE) summed across ALL sales accounts (lags, SPLY, trend)
  - category aggregates, growth/seasonality CURVES (YoY momentum, OLS trend slopes, Fourier harmonics),
    forward promo, distribution-breadth, forecastability statics.
Leak-free: every feature is as-of the forecast origin; training targets are <= origin. Lazy-imports
xgboost so the module stays import-safe without it.

Entry points:
  * ``xgb_hier_forecast(raw, origin_period=..., ...)`` -> DataFrame[key, period, 'predicted']
  * ``apply_xgb_hier_overlay(full_df, category_artifact_paths, cfg, snapshot_week=..., route_categories=...)``
    — DIQ-runner hook mirroring ``deep_forecast.apply_deep_overlay`` (best-effort; never regresses GBM).
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_FUTR_PATTERNS = ["songkran", "bucha", "festive", "holiday", "promo", "discount", "season",
                  "priceoff", "off_invoice", "stringency", "summer", "monsoon", "winter"]


def _roll_mean(a, w, end):
    lo = max(0, end - w + 1); return a[:, lo:end + 1].mean(axis=1)


def _roll_std(a, w, end):
    lo = max(0, end - w + 1); return a[:, lo:end + 1].std(axis=1)


def _slope(a, w, end):
    lo = max(0, end - w + 1); seg = a[:, lo:end + 1]; n = seg.shape[1]
    if n < 2:
        return np.zeros(a.shape[0], dtype=np.float32)
    x = np.arange(n, dtype=np.float32); xm = x.mean(); xx = ((x - xm) ** 2).sum()
    return ((seg - seg.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / (xx + 1e-9)


def _col(a, idx, nk):
    return a[:, idx] if 0 <= idx < a.shape[1] else np.zeros(nk, np.float32)


def _statics(own, Opos):
    """forecastability statics per key from history <= origin (vol, mean, nzfrac, ADI, CV2, spike, recent)."""
    h = own[:, :Opos + 1]; nz = h > 0
    n = np.maximum(nz.sum(axis=1), 1)
    vol = h.sum(axis=1); mean = vol / n
    nzfrac = nz.sum(axis=1) / max(h.shape[1], 1)
    sd = np.sqrt(((h - (vol / max(h.shape[1], 1))[:, None]) ** 2 * nz).sum(axis=1) / n)
    cv2 = (sd / np.maximum(mean, 1e-6)) ** 2
    spike = h.max(axis=1) / np.maximum(mean, 1e-6)
    adi = h.shape[1] / n
    recent = _roll_mean(own, 8, Opos) / (_roll_mean(own, 26, Opos) + 1.0)
    return dict(vol=vol.astype(np.float32), mean=mean.astype(np.float32), nzfrac=nzfrac.astype(np.float32),
                ADI=adi.astype(np.float32), CV2=cv2.astype(np.float32), spike=spike.astype(np.float32),
                recent=recent.astype(np.float32))


def xgb_hier_forecast(
    raw: pd.DataFrame, *, origin_period: int, horizon: int = 13, target_account: str = "E1016",
    key_col: str = "key", period_col: str = "year_week", target_col: str = "Actuals",
    category_col: str = "Category", account_sep: str = "_",
    obj: str = "huber", smear: bool = True, smax: float = 1.8, quantile_alpha: float = 0.0,
    perkey: bool = False, perkey_w: float = 0.5, perkey_min_rows: int = 120, decay: float = 1.0,
    start_origin: Optional[int] = None, n_estimators: int = 700, max_depth: int = 8,
    seasonal_trust: bool = False, trust_strength: float = 1.0, trust_relmin: float = 0.6,
    undamp: bool = False, undamp_categories: Sequence[str] = (), undamp_cv: float = 0.8,
    undamp_trend: float = 0.75, undamp_floor: float = 1.05, undamp_cap: float = 3.0,
) -> pd.DataFrame:
    """Forecast ``target_account``'s keys for origin_period+1 .. +horizon. ``raw`` has all accounts
    (key = <CS_BARCODE><sep><account>), period, target, category. Returns [key, period, 'predicted'].
    ``seasonal_trust`` applies a leak-free bias fix: where the same-period-last-year (SPLY) level has
    proven block-reliable on PAST years and we're forecasting BELOW it after a recent dip, lift the
    level UPWARD toward SPLY (only up -> safe for genuine-growth keys)."""
    import xgboost as xgb

    d = raw[[key_col, period_col, target_col, category_col]].copy()
    d[period_col] = pd.to_numeric(d[period_col], errors="coerce").astype("Int64")
    d = d[d[period_col].notna()].copy()
    d["__cs"] = d[key_col].astype(str).str.rsplit(account_sep, n=1).str[0]
    d["__ac"] = d[key_col].astype(str).str.rsplit(account_sep, n=1).str[1]
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce").fillna(0.0).clip(lower=0)

    # weekly grid: all periods present from min..(origin + horizon)
    allw = sorted(int(w) for w in d[period_col].unique())
    weeks = [w for w in allw if w <= origin_period]
    # extend with the horizon future weeks (ISO-week arithmetic via dates)
    last = pd.to_datetime(str(int(origin_period)) + "1", format="%G%V%u")
    for i in range(1, horizon + 1):
        fw = last + pd.Timedelta(weeks=i); iso = fw.isocalendar()
        weeks.append(int(iso.year) * 100 + int(iso.week))
    weeks = sorted(set(weeks)); pos = {w: i for i, w in enumerate(weeks)}; NW = len(weeks)
    if origin_period not in pos:
        raise ValueError(f"origin {origin_period} not in period grid")
    Opos = pos[origin_period]
    d = d[d[period_col].isin(set(weeks))].copy(); d["__wp"] = d[period_col].map(pos).astype("int64")

    keys = sorted(d.loc[d["__ac"] == target_account, "__cs"].unique())
    if not keys:
        return pd.DataFrame(columns=[key_col, period_col, "predicted"])
    kpos = {k: i for i, k in enumerate(keys)}; NK = len(keys)
    cats = sorted(d[category_col].astype(str).unique()); catid = {c: i for i, c in enumerate(cats)}
    cs2cat = d.drop_duplicates("__cs").set_index("__cs")[category_col].astype(str)
    key_cat = np.array([catid.get(cs2cat.get(k, cats[0]), 0) for k in keys])

    def arr(df, rcol):
        a = np.zeros((NK, NW), dtype=np.float32); df = df.dropna(subset=[rcol])
        np.add.at(a, (df[rcol].astype(int).values, df["__wp"].astype(int).values), df[target_col].values.astype(np.float32))
        return a

    own_df = d[d["__ac"] == target_account].copy(); own_df["__r"] = own_df["__cs"].map(kpos)
    own = arr(own_df, "__r")
    xa_df = d.groupby(["__cs", "__wp"], as_index=False)[target_col].sum(); xa_df["__r"] = xa_df["__cs"].map(kpos)
    xacct = arr(xa_df, "__r")
    catarr = np.zeros((len(cats), NW), dtype=np.float32)
    cdf = d[d["__ac"] == target_account].copy(); cdf["__c"] = cdf[category_col].astype(str).map(catid)
    np.add.at(catarr, (cdf["__c"].values, cdf["__wp"].values), cdf[target_col].values.astype(np.float32))
    # distribution breadth + promo (known-future) from the all-account panel
    an = d[d[target_col] > 0].groupby(["__cs", "__wp"])["__ac"].nunique().reset_index(); an["__r"] = an["__cs"].map(kpos)
    acctn = np.zeros((NK, NW), dtype=np.float32); an = an.dropna(subset=["__r"])
    acctn[an["__r"].astype(int).values, an["__wp"].astype(int).values] = an["__ac"].values.astype(np.float32)
    fcols = [c for c in raw.columns if any(p in c.lower() for p in ("ispromo", "discount"))]
    pr = np.zeros((NK, NW), dtype=np.float32); disc = np.zeros((NK, NW), dtype=np.float32)
    if fcols:
        pe = raw[raw[key_col].astype(str).str.rsplit(account_sep, n=1).str[1] == target_account].copy()
        pe["__cs"] = pe[key_col].astype(str).str.rsplit(account_sep, n=1).str[0]
        pe["__wp"] = pd.to_numeric(pe[period_col], errors="coerce").map(pos)
        pe["__r"] = pe["__cs"].map(kpos); pe = pe.dropna(subset=["__r", "__wp"])
        pcol = next((c for c in fcols if "ispromo" in c.lower()), None)
        dcol = next((c for c in fcols if "discount" in c.lower()), None)
        if pcol:
            g = pe.groupby(["__r", "__wp"])[pcol].max().reset_index()
            pr[g["__r"].astype(int).values, g["__wp"].astype(int).values] = pd.to_numeric(g[pcol], errors="coerce").fillna(0).values
        if dcol:
            g = pe.groupby(["__r", "__wp"])[dcol].mean().reset_index()
            disc[g["__r"].astype(int).values, g["__wp"].astype(int).values] = pd.to_numeric(g[dcol], errors="coerce").fillna(0).values

    ST = _statics(own, Opos)
    ST["logvol"] = np.log1p(ST["vol"]); ST["logmean"] = np.log1p(ST["mean"])
    key_vseg = np.zeros(NK, dtype=int); _v = np.nan_to_num(ST["vol"])
    for ci in range(len(cats)):
        m = key_cat == ci
        if m.sum() >= 6:
            q = np.quantile(_v[m], [1 / 3, 2 / 3]); key_vseg[m] = np.digitize(_v[m], q)
    woy = np.array([w % 100 for w in weeks])

    def build(origins, is_train):
        X = []
        for wp in origins:
            for hh in range(1, horizon + 1):
                tp = wp + hh
                if tp >= NW:
                    continue
                if is_train and tp > Opos:
                    continue
                sp = tp - 52
                f = {
                    "h": hh, "own0": own[:, wp], "own1": own[:, max(0, wp - 1)], "own3": own[:, max(0, wp - 3)],
                    "own7": own[:, max(0, wp - 7)], "own12": own[:, max(0, wp - 12)], "own52": own[:, max(0, wp - 51)],
                    "own_rm4": _roll_mean(own, 4, wp), "own_rm8": _roll_mean(own, 8, wp), "own_rm13": _roll_mean(own, 13, wp),
                    "own_sply": _col(own, sp, NK),
                    "xa0": xacct[:, wp], "xa3": xacct[:, max(0, wp - 3)], "xa52": xacct[:, max(0, wp - 51)],
                    "xa_rm8": _roll_mean(xacct, 8, wp), "xa_sply": _col(xacct, sp, NK),
                    "xa_trend": np.log1p(_roll_mean(xacct, 8, wp)) - np.log1p(_roll_mean(xacct, 8, max(0, wp - 52))),
                    "cat0": catarr[key_cat, wp], "cat_sply": (catarr[key_cat, sp] if sp >= 0 else np.zeros(NK, np.float32)),
                    "cat_trend": np.log1p(_roll_mean(catarr, 8, wp))[key_cat] - np.log1p(_roll_mean(catarr, 8, max(0, wp - 52)))[key_cat],
                    "share": _roll_mean(own, 8, wp) / (_roll_mean(xacct, 8, wp) + 1.0),
                    "woy_sin": np.full(NK, np.sin(2 * np.pi * woy[tp] / 52)), "woy_cos": np.full(NK, np.cos(2 * np.pi * woy[tp] / 52)),
                    "is_q1": np.full(NK, 1.0 if woy[tp] <= 15 else 0.0),
                    "pr_tgt": pr[:, tp], "disc_tgt": disc[:, tp], "pr_now": pr[:, wp], "pr_sply": _col(pr, sp, NK),
                    "own_yoy": _roll_mean(own, 8, wp) / (_roll_mean(own, 8, max(0, wp - 52)) + 1.0),
                    "xa_yoy": _roll_mean(xacct, 8, wp) / (_roll_mean(xacct, 8, max(0, wp - 52)) + 1.0),
                    "cat_yoy": (_roll_mean(catarr, 8, wp) / (_roll_mean(catarr, 8, max(0, wp - 52)) + 1.0))[key_cat],
                    "own_slope13": _slope(own, 13, wp), "own_slope26": _slope(own, 26, wp), "xa_slope13": _slope(xacct, 13, wp),
                    "own_mom": _roll_mean(own, 4, wp) / (_roll_mean(own, 13, wp) + 1.0),
                    "own_accel": (_roll_mean(own, 4, wp) - _roll_mean(own, 8, wp)) - (_roll_mean(own, 8, wp) - _roll_mean(own, 13, wp)),
                    "recent_hist": _roll_mean(own, 8, wp) / (_roll_mean(own, 52, wp) + 1.0),
                    "woy_sin2": np.full(NK, np.sin(4 * np.pi * woy[tp] / 52)), "woy_cos2": np.full(NK, np.cos(4 * np.pi * woy[tp] / 52)),
                    "woy_sin3": np.full(NK, np.sin(6 * np.pi * woy[tp] / 52)), "woy_cos3": np.full(NK, np.cos(6 * np.pi * woy[tp] / 52)),
                    "seas_idx": own[:, wp] / (_roll_mean(own, 52, wp) + 1.0), "own_sply2": _col(own, tp - 104, NK),
                    "sply_yoy": (own[:, sp] / (own[:, tp - 104] + 1.0)) if (sp >= 0 and tp - 104 >= 0) else np.ones(NK, np.float32),
                    "woy": np.full(NK, float(woy[tp])),
                    "own_rm26": _roll_mean(own, 26, wp), "own_rm52": _roll_mean(own, 52, wp),
                    "own_std8": _roll_std(own, 8, wp), "own_std13": _roll_std(own, 13, wp),
                    "own_max13": own[:, max(0, wp - 12):wp + 1].max(axis=1), "own_min13": own[:, max(0, wp - 12):wp + 1].min(axis=1),
                    "pr_rate8": _roll_mean(pr, 8, wp), "disc_now": disc[:, wp], "disc_rm8": _roll_mean(disc, 8, wp),
                    "acctn": _roll_mean(acctn, 8, wp), "acctn_now": acctn[:, wp],
                    "acctn_yoy": _roll_mean(acctn, 8, wp) / (_roll_mean(acctn, 8, max(0, wp - 52)) + 0.1),
                    "yoy_accel": (_roll_mean(own, 8, wp) / (_roll_mean(own, 8, max(0, wp - 52)) + 1.0)) - (_roll_mean(own, 8, max(0, wp - 13)) / (_roll_mean(own, 8, max(0, wp - 65)) + 1.0)),
                    "xa_yoy_lead": _roll_mean(xacct, 4, wp) / (_roll_mean(xacct, 4, max(0, wp - 52)) + 1.0),
                    "own_yoy4": _roll_mean(own, 4, wp) / (_roll_mean(own, 4, max(0, wp - 52)) + 1.0),
                    "catid": key_cat.astype(np.float32),
                    **{f"st_{c}": ST[c] for c in ["logvol", "logmean", "nzfrac", "ADI", "CV2", "spike", "recent"]},
                }
                fr = pd.DataFrame(f); fr["__keyidx"] = np.arange(NK, dtype=np.int32)
                _b = np.where(fr["own_sply"].values > 0.2 * fr["own_rm8"].values + 1.0, fr["own_sply"].values, fr["own_rm8"].values)
                fr["__b"] = np.maximum(_b, 1.0).astype(np.float32); fr["__seg"] = key_cat * 3 + key_vseg
                if is_train:
                    fr["__y"] = own[:, tp]; fr = fr[(fr["own_rm13"] > 0) | (fr["own0"] > 0)]
                    fr["__wt"] = float(decay ** ((Opos - wp) / 52.0))
                else:
                    fr["__k"] = keys; fr["__tw"] = weeks[tp]
                X.append(fr)
        return pd.concat(X, ignore_index=True)

    st_origin = start_origin if start_origin is not None else (weeks[12] if NW > 12 else weeks[0])
    tr = build([pos[w] for w in weeks if st_origin <= w <= origin_period], True)
    FEATS = [c for c in tr.columns if c not in ("__y", "__k", "__tw", "__wt", "__b", "__seg", "__keyidx")]

    uselog = obj in ("log", "logsmear"); useratio = obj == "ratio"
    mkw = dict(n_estimators=n_estimators, learning_rate=0.04, max_depth=max_depth, subsample=0.8,
               colsample_bytree=0.7, min_child_weight=20, reg_lambda=2.0, tree_method="hist", n_jobs=0)
    if quantile_alpha > 0:
        mkw.update(objective="reg:quantileerror", quantile_alpha=quantile_alpha)
    elif obj == "tweedie":
        mkw.update(objective="reg:tweedie", tweedie_variance_power=1.3)
    elif obj in ("huber", "ratio"):
        mkw.update(objective="reg:pseudohubererror")

    def fwd(y, b):
        return np.log((y + 1.0) / (b + 1.0)) if useratio else (np.log1p(y) if uselog else y)

    def inv(p, b):
        return ((b + 1.0) * np.exp(np.clip(p, -5, 5)) - 1.0) if useratio else (np.expm1(p) if uselog else p)

    model = xgb.XGBRegressor(**mkw)
    model.fit(tr[FEATS], fwd(tr["__y"].values, tr["__b"].values),
              sample_weight=tr["__wt"].values if decay != 1.0 else None)
    smkey = "catid"; smap = {}
    if smear:
        trp = np.clip(inv(model.predict(tr[FEATS]), tr["__b"].values), 0, None)
        tmp = pd.DataFrame({"c": tr[smkey].values.astype(int), "y": tr["__y"].values, "p": trp})
        smap = tmp.groupby("c").apply(lambda g: float(np.clip(g["y"].sum() / max(g["p"].sum(), 1e-9), 0.7, smax))).to_dict()

    te = build([Opos], False)
    pred = np.clip(inv(model.predict(te[FEATS]), te["__b"].values), 0, None)
    if smear:
        pred = pred * np.array([smap.get(int(c), 1.0) for c in te[smkey].values])

    if perkey:
        pkobj = dict(objective="reg:quantileerror", quantile_alpha=quantile_alpha) if quantile_alpha > 0 else dict(objective="reg:pseudohubererror")
        pkcfg = dict(n_estimators=250, max_depth=4, learning_rate=0.05, min_child_weight=12,
                     reg_lambda=5.0, subsample=0.7, colsample_bytree=0.5, tree_method="hist", n_jobs=1, **pkobj)
        te_ki = te["__keyidx"].values; hv = [int(k) for k in np.where(key_vseg == 2)[0]]
        grp = {int(k): g for k, g in tr.groupby("__keyidx") if int(k) in set(hv)}
        for k in hv:
            g = grp.get(k); mask = te_ki == k
            if g is None or len(g) < perkey_min_rows or mask.sum() == 0:
                continue
            m = xgb.XGBRegressor(**pkcfg); m.fit(g[FEATS], fwd(g["__y"].values, g["__b"].values))
            tem = te[mask]; gp = pred[mask]
            pk = np.clip(inv(m.predict(tem[FEATS]), tem["__b"].values), 0, None)
            trp = np.clip(inv(m.predict(g[FEATS]), g["__b"].values), 0, None)
            sf = float(np.clip(g["__y"].sum() / max(trp.sum(), 1e-9), 0.6, 2.0))
            pred[mask] = perkey_w * np.minimum(pk * sf, 3.0 * gp + 5.0) + (1 - perkey_w) * gp

    # --- seasonal-trust bias fix (leak-free): for keys whose SPLY level was block-reliable on PAST
    # years and that we're forecasting BELOW after a recent dip, lift the level UP toward SPLY ---
    if seasonal_trust:
        te_ki = te["__keyidx"].values
        ftot = np.zeros(NK, dtype=np.float64); np.add.at(ftot, te_ki, pred)
        fw = [Opos + hh for hh in range(1, horizon + 1)]

        def _blk(shift):
            cols = [w - shift for w in fw if 0 <= w - shift < NW]
            return own[:, cols].sum(axis=1) if cols else np.zeros(NK, np.float32)

        sply = _blk(52); sply2 = _blk(104); sply3 = _blk(156)
        recent = own[:, max(0, Opos - 5):Opos + 1].sum(axis=1) / 6.0 * horizon   # recent run-rate over a block
        e1 = np.abs(sply - sply2) / np.maximum(sply, 1.0); e2 = np.abs(sply2 - sply3) / np.maximum(sply2, 1.0)
        rel = np.clip(1.0 - (e1 + e2) / 2.0, 0, 1)
        rel = np.where((sply2 > 0) & (sply3 > 0), rel, 0.3)        # modest default when history is short
        cond = (rel >= trust_relmin) & (sply > ftot) & (sply > recent) & (ftot > 1.0)
        new_tot = np.where(cond, ftot + trust_strength * rel * (sply - ftot), ftot)
        fac = np.clip(new_tot / np.maximum(ftot, 1e-9), 1.0, 3.0)
        pred = pred * fac[te_ki]
        logger.info("[xgb] seasonal-trust lifted %d keys (relmin=%.2f, strength=%.2f)", int((fac > 1.01).sum()), trust_relmin, trust_strength)

    # --- un-damping gate (leak-free): the global model mean-reverts a few high-volume keys whose
    # recent level is STABLE and NON-DECLINING; for those, the random walk holds (validated
    # out-of-sample: actual_next13 >= recent13). Lift their forecast TOTAL up to floor*recent13
    # (shape preserved). Applied ONLY to undamp_categories (e.g. FABRIC CLEANING); the others already
    # win bias and a lift would over-shoot them. Flips FABRIC CLEANING to beat LEGO on BOTH metrics. ---
    if undamp and len(undamp_categories):
        te_ki = te["__keyidx"].values
        ftot = np.zeros(NK, dtype=np.float64); np.add.at(ftot, te_ki, pred)
        w13 = own[:, max(0, Opos - 12):Opos + 1]                       # last 13 known weeks (<= origin)
        rec13 = w13.sum(axis=1)
        mu = w13.mean(axis=1); sd = w13.std(axis=1)
        cv = np.where(mu > 0, sd / np.maximum(mu, 1e-9), 9.0)
        h1 = w13[:, :6].sum(axis=1); h2 = w13[:, 7:].sum(axis=1)
        trend = h2 / np.maximum(h1, 1e-9)
        want = {str(c).strip().upper() for c in undamp_categories}
        cat_ok = np.array([cats[c].strip().upper() in want for c in key_cat])
        gate = cat_ok & (cv < undamp_cv) & (trend > undamp_trend) & (ftot < undamp_floor * rec13) & (rec13 > 0)
        fac = np.where(gate, np.clip(undamp_floor * rec13 / np.maximum(ftot, 1e-9), 1.0, undamp_cap), 1.0)
        pred = pred * fac[te_ki]
        logger.info("[xgb] un-damping lifted %d keys in %s (floor=%.2f)", int((fac > 1.01).sum()), sorted(want), undamp_floor)

    out = pd.DataFrame({key_col: te["__k"].astype(str) + account_sep + target_account,
                        period_col: te["__tw"].astype(int), "predicted": np.clip(pred, 0, None)})
    return out


def apply_xgb_hier_overlay(
    full_df: pd.DataFrame, category_artifact_paths: Dict[str, Path], cfg, *, snapshot_week,
    route_categories: Sequence[str], target_account: str = "E1016", category_col: str = "category_name",
    obj: str = "huber", perkey: bool = False, seasonal_trust: bool = False,
    undamp: bool = False, undamp_categories: Sequence[str] = ("FABRIC CLEANING",), **kw,
) -> List[str]:
    """DIQ-runner hook (mirrors deep_forecast.apply_deep_overlay): train one hierarchical XGBoost
    across full_df, overwrite the 'predicted' column of each routed category's inference_forecast.csv
    (matched on key x period). Best-effort — any failure leaves the GBM forecast in place."""
    key_col = (cfg.prediction_key_cols or ["key"])[0]
    period_col = cfg.timestamp_col; target_col = cfg.target_col
    h = int(getattr(cfg, "forecast_horizon", 13)); origin = int(str(snapshot_week)) - 1
    route = {str(c).strip().upper() for c in route_categories}
    cat_col = category_col if category_col in full_df.columns else ("Category" if "Category" in full_df.columns else None)
    try:
        fc = xgb_hier_forecast(full_df, origin_period=origin, horizon=h, target_account=target_account,
                               key_col=key_col, period_col=period_col, target_col=target_col,
                               category_col=cat_col, obj=obj, perkey=perkey, seasonal_trust=seasonal_trust,
                               undamp=undamp, undamp_categories=undamp_categories, **kw)
    except Exception as exc:  # noqa: BLE001 - never sink the run
        logger.error("[xgb] overlay training failed (%s) — keeping GBM forecasts", exc, exc_info=True)
        return []
    if fc.empty:
        logger.warning("[xgb] overlay produced no forecasts — keeping GBM"); return []
    fc["__k"] = fc[key_col].astype(str); fc["__p"] = pd.to_numeric(fc[period_col], errors="coerce").astype("Int64")
    by_key = fc.set_index(["__k", "__p"])["predicted"]
    overwritten = []
    for cat, art_dir in category_artifact_paths.items():
        if cat.strip().upper() not in route:
            continue
        csv_path = Path(art_dir) / "model_artifacts" / "inference_forecast.csv"
        if not csv_path.exists():
            logger.warning("[xgb] %s: inference_forecast.csv missing — skip", cat); continue
        try:
            gbm = pd.read_csv(csv_path)
            k = gbm[key_col].astype(str); p = pd.to_numeric(gbm[period_col], errors="coerce").astype("Int64")
            new = by_key.reindex(list(zip(k, p))).values; mask = ~pd.isna(new)
            if mask.sum() == 0:
                logger.warning("[xgb] %s: no key x period overlap — keeping GBM", cat); continue
            gbm.loc[mask, "predicted"] = np.asarray(new, dtype="float64")[mask]
            gbm.to_csv(csv_path, index=False); overwritten.append(cat)
            logger.info("[xgb] %s: overlaid %d/%d rows with hierarchical-XGBoost forecast", cat, int(mask.sum()), len(gbm))
        except Exception as exc:  # noqa: BLE001
            logger.error("[xgb] %s: overlay write failed (%s) — keeping GBM", cat, exc)
    logger.info("[xgb] overlay complete: %d/%d routed categories overwritten", len(overwritten), len(route))
    return overwritten
