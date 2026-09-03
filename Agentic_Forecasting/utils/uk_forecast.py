"""
UK deterministic direct-multi-horizon forecaster (local-first, config-driven).

Design (driven by the Phase-2 diagnosis):
  * Per-horizon GBMs h=1..H. Horizon h is trained to map features-known-at-week-w
    to actual_sales at week w+h. No recursion => no error accumulation, and each
    head is tuned for its exact lag (we care about lag 4 & 5).
  * Tweedie objective => zero-inflated/intermittent-friendly. The benchmark
    `prediction_rf` has NEGATIVE accuracy on intermittent/lumpy keys (it over-
    forecasts sparse demand); a tweedie head + recency weighting shrinks there.
  * Pooled GLOBAL model across all in-scope categories with category/brand/segment
    as categorical features => one model covers all 13 categories, including the
    12 with no prior model, and shares signal.
  * Bias levers: SPLY seasonal anchoring (utils/seasonal_blend, the proven lever)
    + optional out-of-sample bias calibration.

Leakage safety:
  * Backward features (lags/rolling) at week w use only data <= w.
  * Forward covariates (promo/holiday/season/weather) are taken at the TARGET week
    w+h — these are genuinely known in advance.
  * Target weeks for training are <= train_end (= snapshot-1). Forecast origin is
    train_end, so step h lands on year_week = snapshot+(h-1).
  * Deterministic: fixed seeds + LightGBM deterministic=True.

Entry point: forecast_origin(snapshot, cfg) -> DataFrame[key, year_week, forecast, category_name, horizon].
"""
from __future__ import annotations

import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from utils import seasonal_blend

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("datacache/uk")
BASE_CONFIG = "config/config_uk_base_local.yaml"


# --------------------------------------------------------------------------- #
# config + week arithmetic
# --------------------------------------------------------------------------- #
def load_config(path: str = BASE_CONFIG, overrides: Optional[dict] = None) -> dict:
    # Databricks pipeline needs .py config: load_config_file prefers the config_*_base_local.py
    # sibling of `path` and falls back to the legacy .yaml (Thailand / others).
    from utils.config_loader import load_config_file
    cfg = load_config_file(path)
    if overrides:
        cfg.update(overrides)
    return cfg


def add_weeks(yyyyww: str, n: int) -> str:
    """ISO-correct week add (handles 52/53-week years). Accepts both compact
    '202618' and hyphenated '2026-18' (the Databricks widget form)."""
    s = str(yyyyww).replace("-", "")
    y, w = int(s[:4]), int(s[4:])
    monday = datetime.strptime(f"{y}-W{w:02d}-1", "%G-W%V-%u")
    iso = (monday + pd.Timedelta(weeks=n)).isocalendar()
    return f"{iso[0]}{iso[1]:02d}"


# --------------------------------------------------------------------------- #
# input panel
# --------------------------------------------------------------------------- #
def read_input_panel(snapshot: str, source: str = "trim") -> pd.DataFrame:
    """source: 'trim' (88-col curated cache) or 'full' (all 294 cols, input_full/)."""
    sub = "input_full" if source == "full" else "input"
    parts = sorted(glob.glob(str(CACHE_ROOT / sub / f"snapshot={snapshot}" / "part-*.parquet")))
    if not parts:
        raise FileNotFoundError(
            f"No cached {sub} for snapshot {snapshot}. Run: "
            f".venv/bin/python scripts/uc_cache.py --kind input {'--full ' if source=='full' else ''}"
            f"--snapshots {snapshot[:4]}-{snapshot[4:]}"
        )
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    df["year_week"] = df["year_week"].astype(str)
    return df


def _future_cols(df: pd.DataFrame, cfg: dict) -> List[str]:
    pref = tuple(cfg["future_cov_prefixes"])
    drop = {cfg["target"], cfg["key_col"], cfg["time_col"], cfg["cat_col"], "exclude_forecast"}
    return [c for c in df.columns if c.startswith(pref) and c not in drop
            and pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------- #
# feature engineering (backward, leakage-safe)
# --------------------------------------------------------------------------- #
def _grp_roll(shifted: pd.Series, keys: pd.Series, window: int, how: str) -> pd.Series:
    r = shifted.groupby(keys, sort=False).rolling(window, min_periods=1 if how == "mean" else 2)
    out = (r.mean() if how == "mean" else r.std()).reset_index(level=0, drop=True)
    return out


def add_backward_features(df: pd.DataFrame, cfg: dict) -> List[str]:
    """Add lag / rolling / sparsity features (computed as of each week w). Returns
    the list of backward feature column names. Mutates df (sorted by key, week)."""
    key, tgt = cfg["key_col"], cfg["target"]
    df.sort_values([key, cfg["time_col"]], inplace=True, kind="mergesort")
    g = df.groupby(key, sort=False)
    cols: List[str] = []

    for L in cfg["lags"]:
        c = f"lag_{L}"
        df[c] = g[tgt].shift(L)
        cols.append(c)

    for sig in cfg.get("extra_signal_cols", []):
        if sig in df.columns:
            for L in (1, 4, 13):
                c = f"{sig}_lag{L}"
                df[c] = g[sig].shift(L)
                cols.append(c)

    shifted = g[tgt].shift(1)
    for W in cfg["roll_windows"]:
        cm, cs = f"rmean_{W}", f"rstd_{W}"
        df[cm] = _grp_roll(shifted, df[key], W, "mean")
        df[cs] = _grp_roll(shifted, df[key], W, "std")
        cols += [cm, cs]

    # sparsity / trend signals (the diagnosis: sparse demand is the opening)
    nz = (g[tgt].shift(1) > 0).astype("float64")
    df["zerofrac_26"] = 1.0 - _grp_roll(nz, df[key], 26, "mean")
    df["sply_ratio"] = (df.get("lag_1", 0) + 1.0) / (df.get("lag_52", 0).fillna(0) + 1.0) if "lag_52" in df else np.nan
    cols += ["zerofrac_26"] + (["sply_ratio"] if "sply_ratio" in df else [])

    # Hierarchical aggregate features: category- and APG-group lagged demand trend
    # (shared signal that helps sparse keys; the xgb_hier lever). Mapped back IN PLACE
    # (a merge would rebind the local df and lose the columns for the caller).
    if cfg.get("rich_features", False):
        twk = cfg["time_col"]
        for gcol, pref in [(cfg["cat_col"], "cat"), ("APG_code", "apg")]:
            if gcol in df.columns:
                gw = df.groupby([gcol, twk], as_index=False)[tgt].sum().rename(columns={tgt: "_tot"})
                gw = gw.sort_values([gcol, twk])
                for W in (4, 13):
                    gw[f"{pref}_roll{W}"] = (gw.groupby(gcol)["_tot"]
                                             .transform(lambda s: s.shift(1).rolling(W, min_periods=1).mean()))
                gw = gw.set_index([gcol, twk])
                gidx = pd.MultiIndex.from_arrays([df[gcol], df[twk]])
                for W in (4, 13):
                    df[f"{pref}_roll{W}"] = gw[f"{pref}_roll{W}"].reindex(gidx).to_numpy()
                cols += [f"{pref}_roll4", f"{pref}_roll13"]
    return cols


def _encode_categoricals(df: pd.DataFrame, cfg: dict) -> List[str]:
    """Label-encode static categorical features in place. Returns kept cat cols."""
    out = []
    for c in cfg.get("cat_features", []):
        if c in df.columns:
            # +1 so NaN's -1 code becomes 0 (LightGBM categoricals must be non-negative).
            df[c] = (df[c].astype("category").cat.codes + 1).astype("int32")
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# train + predict one origin
# --------------------------------------------------------------------------- #
def _lgb_params(cfg: dict) -> dict:
    obj = cfg["objective"]
    p = dict(
        objective=obj,
        num_leaves=cfg["num_leaves"],
        learning_rate=cfg["learning_rate"],
        min_data_in_leaf=cfg["min_data_in_leaf"],
        lambda_l2=cfg["lambda_l2"],
        feature_fraction=cfg["feature_fraction"],
        bagging_fraction=cfg["bagging_fraction"],
        bagging_freq=cfg.get("bagging_freq", 1),
        max_depth=cfg.get("max_depth", -1),
        num_threads=cfg.get("num_threads", 0),
        seed=cfg["seed"], bagging_seed=cfg["seed"], feature_fraction_seed=cfg["seed"],
        deterministic=True, force_row_wise=True, verbosity=-1,
    )
    # objective-specific params (WAPE is L1-like, so quantile@~0.5 / regression_l1 align with the metric)
    if obj == "tweedie":
        p["tweedie_variance_power"] = cfg.get("tweedie_variance_power", 1.3)
    elif obj == "quantile":
        p["alpha"] = cfg.get("alpha", 0.5)
    return p


def _forecast_at(df, cutoff, horizons, *, left, right, feat_cols, cat_cols, weeks, rank, cfg, key, twk, tag):
    """Train per-horizon models on target weeks <= cutoff and predict from origin=cutoff.
    Step h lands on year_week = cutoff + h. Returns [key, year_week, forecast_raw, horizon]."""
    import lightgbm as lgb
    te_rank = rank.get(cutoff, max(rank.values()))
    out_frames = []
    for h in horizons:
        tw_map = {w: add_weeks(w, h) for w in weeks}
        left_h = left.copy()
        left_h["_tw"] = left_h[twk].map(tw_map)
        pairs = left_h.merge(right, on=[key, "_tw"], how="inner")
        tr = pairs[pairs["_tw"] <= cutoff].copy()
        if cfg.get("train_sample_frac", 1.0) < 1.0 and len(tr):
            tr = tr.sample(frac=cfg["train_sample_frac"], random_state=cfg["seed"])
        pr = pairs[pairs[twk] == cutoff].copy()
        if tr.empty or pr.empty:
            logger.warning("[uk_fc] %s cutoff %s h=%d: train=%d pred=%d (skipped)", tag, cutoff, h, len(tr), len(pr))
            continue
        wt = np.power(0.5, (te_rank - tr["_wi"]).clip(lower=0) / cfg["recency_halflife_weeks"])
        dtrain = lgb.Dataset(tr[feat_cols], label=tr["_y"].clip(lower=0), weight=wt.values,
                             categorical_feature=cat_cols, free_raw_data=False)
        model = lgb.train(_lgb_params(cfg), dtrain, num_boost_round=cfg["n_estimators"])
        yhat = model.predict(pr[feat_cols])
        res = pr[[key, "_tw"]].rename(columns={"_tw": twk}).copy()
        res["forecast_raw"] = np.clip(yhat, 0, None) if cfg.get("clip_negatives", True) else yhat
        res["horizon"] = h
        out_frames.append(res)
        logger.info("[uk_fc] %s cutoff %s h=%2d: trained on %s rows -> %s forecasts",
                    tag, cutoff, h, f"{len(tr):,}", f"{len(res):,}")
    if not out_frames:
        return pd.DataFrame(columns=[key, twk, "forecast_raw", "horizon"])
    return pd.concat(out_frames, ignore_index=True)


def forecast_origin(
    snapshot: str,
    cfg: dict,
    *,
    horizons: Optional[Sequence[int]] = None,
    panel: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Train per-horizon models on history <= snapshot-1 and forecast the window.
    Returns DataFrame[key, year_week, category_name, horizon, forecast_raw, forecast].

    Levers: (1) SPLY seasonal anchoring (utils/seasonal_blend), (2) optional honest
    out-of-sample per-category bias calibration (a held-out forecast from an earlier
    origin learns the residual bias factor; no leakage)."""
    key, tgt, twk, catc = cfg["key_col"], cfg["target"], cfg["time_col"], cfg["cat_col"]
    train_end = add_weeks(snapshot, -1)
    H = cfg["horizons"]
    horizons = list(horizons) if horizons is not None else list(range(1, H + 1))

    df = panel if panel is not None else read_input_panel(snapshot, source=cfg.get("input_source", "trim"))
    df = df.copy()
    df[tgt] = pd.to_numeric(df[tgt], errors="coerce").fillna(0.0)

    # Collapse the promo one-hot families into proper categoricals (only useful in
    # 'full' mode where the families + string-list fields are present). Adds 5-9 new
    # categorical columns (promo_mechanic, promo_feature, promo_objective, ...).
    if cfg.get("collapse_promo", False):
        from utils.uk_features import collapse_promo_groups
        df, extra_cats = collapse_promo_groups(df, drop_onehots=cfg.get("drop_promo_onehots", True))
        # ensure these get encoded as categoricals
        cur = list(cfg.get("cat_features", []))
        for c in extra_cats:
            if c not in cur:
                cur.append(c)
        cfg = {**cfg, "cat_features": cur}

    # Signed weeks-to-holiday distance features (Easter / May Day / Spring Bank / ...).
    # Known in advance => valid future-covariates that fix the moving-holiday blind spot
    # of our rolling-mean-dominated model. Added BEFORE _future_cols so the 'weeks_to_'
    # prefix picks them up automatically.
    if cfg.get("holiday_distance_features", False):
        from utils.uk_holidays import add_holiday_features
        added = add_holiday_features(df, yw_col=twk)
        logger.info("[uk_fc] added %d holiday-distance features: %s", len(added), added)

    # Capture RAW category names before _encode_categoricals turns category_name into
    # int codes (the model feature). Calibration + output labelling need real names.
    catmap = df.drop_duplicates(key).set_index(key)[catc]

    fut_cols = _future_cols(df, cfg)
    back_cols = add_backward_features(df, cfg)
    cat_cols = _encode_categoricals(df, cfg)
    left_feats = back_cols + cat_cols                       # known at w
    feat_cols = left_feats + fut_cols                       # + forward covs at w+h

    weeks = sorted(df[twk].unique())
    rank = {w: i for i, w in enumerate(weeks)}
    df["_wi"] = df[twk].map(rank)
    left = df[[key, twk, "_wi"] + left_feats].copy()
    right = df[[key, twk] + fut_cols + [tgt]].rename(columns={twk: "_tw", tgt: "_y"})
    common = dict(left=left, right=right, feat_cols=feat_cols, cat_cols=cat_cols,
                  weeks=weeks, rank=rank, cfg=cfg, key=key, twk=twk)

    def _post(fc, cutoff):
        """Attach category + apply SPLY seasonal anchoring (history <= cutoff)."""
        if fc.empty:
            return fc
        fc = fc.copy()
        fc[catc] = fc[key].map(catmap)
        fc["forecast"] = fc["forecast_raw"]
        if cfg.get("seasonal_anchor", "none") != "none":
            hist = df[df[twk] <= cutoff][[key, twk, tgt, catc]].copy()
            fc = seasonal_blend.seasonal_correct(
                fc, hist, key_col=key, date_col=twk, target_col=tgt, pred_col="forecast",
                cat_col=catc, level=cfg["seasonal_anchor"], lo=cfg.get("seasonal_lo", 0.5),
                hi=cfg.get("seasonal_hi", 2.0))
            if cfg.get("clip_negatives", True):
                fc["forecast"] = fc["forecast"].clip(lower=0)
        return fc

    def _run_at(cutoff, tag):
        """Global model by default; if per_category, train one model per category
        (optionally restricted to per_category_categories) and concat the forecasts."""
        if not cfg.get("per_category", False):
            return _forecast_at(df, cutoff, horizons, tag=tag, **common)
        only = set(cfg.get("per_category_categories") or [])   # empty => every category
        parts, glob_keys = [], set()
        for cat in catmap.dropna().unique():
            if only and cat not in only:
                continue
            ckeys = set(catmap.index[catmap == cat])
            glob_keys |= ckeys
            lc = common["left"][common["left"][key].isin(ckeys)]
            rc = common["right"][common["right"][key].isin(ckeys)]
            if not lc.empty and not rc.empty:
                parts.append(_forecast_at(df, cutoff, horizons, tag=f"{tag}:{str(cat)[:8]}",
                                          **{**common, "left": lc, "right": rc}))
        if only:   # categories not in `only` keep the global model
            rest_l = common["left"][~common["left"][key].isin(glob_keys)]
            rest_r = common["right"][~common["right"][key].isin(glob_keys)]
            if not rest_l.empty:
                parts.append(_forecast_at(df, cutoff, horizons, tag=f"{tag}:global",
                                          **{**common, "left": rest_l, "right": rest_r}))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=[key, twk, "forecast_raw", "horizon"])

    # --- (2) honest out-of-sample bias calibration: forecast an earlier origin,
    #     compare to KNOWN actuals, learn a per-category factor, apply to the real run.
    factors = None
    if cfg.get("bias_calibration", False):
        V = cfg.get("bias_calibration_holdout_weeks", 6)
        calib_end = add_weeks(train_end, -V)
        cfc = _post(_run_at(calib_end, "calib"), calib_end)
        if not cfc.empty:
            act = df[[key, twk, tgt]].copy()
            act[twk] = act[twk].astype(str)
            cfc[twk] = cfc[twk].astype(str)
            cfc = cfc.merge(act.rename(columns={tgt: "_a"}), on=[key, twk], how="left")
            cfc["_a"] = cfc["_a"].fillna(0.0)
            agg = cfc.groupby(catc).agg(a=("_a", "sum"), p=("forecast", "sum"))
            raw = agg["a"] / agg["p"].clip(lower=1e-9)
            # Damp toward 1.0 (the calib-origin factor is noisy; a full shift over-corrects).
            damp = cfg.get("bias_calibration_damp", 1.0)
            factors = (1.0 + damp * (raw - 1.0)).clip(
                cfg.get("bias_factor_min", 0.5), cfg.get("bias_factor_max", 1.5))
            # Selective: only calibrate the listed categories (others stay at 1.0).
            only = cfg.get("calibrate_categories")
            if only:
                factors = factors.where(factors.index.isin(set(only)), 1.0)
            logger.info("[uk_fc] %s bias factors: %s", snapshot, {k: round(v, 3) for k, v in factors.items()})

    fc = _post(_run_at(train_end, "final"), train_end)
    if fc.empty:
        return pd.DataFrame(columns=[key, twk, catc, "horizon", "forecast_raw", "forecast"])
    if factors is not None:
        fc["forecast"] = fc["forecast"] * fc[catc].map(factors).fillna(1.0)
        if cfg.get("clip_negatives", True):
            fc["forecast"] = fc["forecast"].clip(lower=0)
    return fc


# --------------------------------------------------------------------------- #
# Hierarchical stacked ensemble (global + per-category, blended per category)
# --------------------------------------------------------------------------- #
def _learn_blend_weights(valg: pd.DataFrame, cat_col: str) -> Dict[str, float]:
    """Per-category convex weight w in [0,1] (local share) that minimises GTIN-level
    WAPE of w*local + (1-w)*global on the validation frame. w->1 = trust local,
    w->0 = trust global. A 21-point grid (WAPE is piecewise-linear convex in w)."""
    ws = np.linspace(0.0, 1.0, 21)
    out: Dict[str, float] = {}
    for cat, gg in valg.groupby(cat_col):
        A, G, L = gg["actual"].to_numpy(), gg["g"].to_numpy(), gg["l"].to_numpy()
        sa = A.sum()
        if sa <= 0:
            out[cat] = 0.0
            continue
        wapes = [np.abs((w * L + (1 - w) * G) - A).sum() / sa for w in ws]
        out[cat] = float(ws[int(np.argmin(wapes))])
    return out


def forecast_origin_stacked(snapshot, cfg, *, eval_lags=(4, 5), panel=None):
    """Hierarchical stacked ensemble. Trains a GLOBAL (pooled) and a LOCAL (per-category)
    base model, then blends them per category with a convex weight learned on an
    out-of-sample validation origin (train_end - stack_val_weeks) — leakage-free, since
    that origin's lag-4/5 weeks have known actuals. Returns (forecast_df, weights)."""
    key, tgt, twk, catc = cfg["key_col"], cfg["target"], cfg["time_col"], cfg["cat_col"]
    train_end = add_weeks(snapshot, -1)
    val_end = add_weeks(train_end, -cfg.get("stack_val_weeks", 6))
    lags = list(eval_lags)

    df = panel if panel is not None else read_input_panel(snapshot)
    df = df.copy()
    df[tgt] = pd.to_numeric(df[tgt], errors="coerce").fillna(0.0)
    catmap = df.drop_duplicates(key).set_index(key)[catc]
    fut_cols = _future_cols(df, cfg)
    back_cols = add_backward_features(df, cfg)
    cat_cols = _encode_categoricals(df, cfg)
    feat_cols = back_cols + cat_cols + fut_cols
    weeks = sorted(df[twk].unique())
    rank = {w: i for i, w in enumerate(weeks)}
    df["_wi"] = df[twk].map(rank)
    left = df[[key, twk, "_wi"] + back_cols + cat_cols].copy()
    right = df[[key, twk] + fut_cols + [tgt]].rename(columns={twk: "_tw", tgt: "_y"})
    common = dict(left=left, right=right, feat_cols=feat_cols, cat_cols=cat_cols,
                  weeks=weeks, rank=rank, cfg=cfg, key=key, twk=twk)

    def _global(cutoff, tag):
        return _forecast_at(df, cutoff, lags, tag=tag, **common)

    def _local(cutoff, tag):
        parts = []
        for cat in catmap.dropna().unique():
            ck = set(catmap.index[catmap == cat])
            lc, rc = left[left[key].isin(ck)], right[right[key].isin(ck)]
            if not lc.empty and not rc.empty:
                parts.append(_forecast_at(df, cutoff, lags, tag=f"{tag}:{str(cat)[:5]}",
                                          **{**common, "left": lc, "right": rc}))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[key, twk, "forecast_raw", "horizon"])

    def _seas(fc, cutoff):
        if fc.empty:
            return fc
        fc = fc.copy()
        fc[catc] = fc[key].map(catmap)
        fc["forecast"] = fc["forecast_raw"]
        if cfg.get("seasonal_anchor", "none") != "none":
            hist = df[df[twk] <= cutoff][[key, twk, tgt, catc]].copy()
            fc = seasonal_blend.seasonal_correct(fc, hist, key_col=key, date_col=twk, target_col=tgt,
                                                 pred_col="forecast", cat_col=catc, level=cfg["seasonal_anchor"],
                                                 lo=cfg.get("seasonal_lo", 0.5), hi=cfg.get("seasonal_hi", 2.0))
        return fc[[key, twk, "horizon", catc, "forecast"]]

    gV, lV = _seas(_global(val_end, "gV"), val_end), _seas(_local(val_end, "lV"), val_end)
    gF, lF = _seas(_global(train_end, "gF"), train_end), _seas(_local(train_end, "lF"), train_end)

    # learn per-category weights on the validation origin (GTIN grain, known actuals)
    val = gV.rename(columns={"forecast": "g"}).merge(
        lV.rename(columns={"forecast": "l"})[[key, twk, "horizon", "l"]], on=[key, twk, "horizon"])
    act = df[[key, twk, tgt]].rename(columns={tgt: "actual"})
    act[twk] = act[twk].astype(str)
    val[twk] = val[twk].astype(str)
    val = val.merge(act, on=[key, twk], how="left")
    val["actual"] = val["actual"].fillna(0.0)
    val["cs_gtin"] = val[key].astype(str).str.rsplit("_", n=1).str[0]
    valg = val.groupby([catc, "cs_gtin", twk], as_index=False).agg(
        g=("g", "sum"), l=("l", "sum"), actual=("actual", "sum"))
    wmap = _learn_blend_weights(valg, catc)
    logger.info("[uk_fc] %s stack local-share: %s", snapshot, {k: round(v, 2) for k, v in wmap.items()})

    # apply learned weights to the real-origin bases
    fin = gF.rename(columns={"forecast": "g"}).merge(
        lF.rename(columns={"forecast": "l"})[[key, twk, "horizon", "l"]], on=[key, twk, "horizon"])
    w = fin[catc].map(wmap).fillna(0.0)
    fin["forecast"] = w * fin["l"] + (1.0 - w) * fin["g"]
    if cfg.get("clip_negatives", True):
        fin["forecast"] = fin["forecast"].clip(lower=0)
    return fin[[key, twk, catc, "horizon", "g", "l", "forecast"]], wmap


def aggregate_to_gtin(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aggregate a key (gtin×customer) panel to the GTIN×week grain — the grain we're
    judged on. Sums non-excluded keys' volume, means rates/prices, maxes flags, keeps
    hierarchy. Returns a panel with key == cs_gtin, ready for forecast_origin(panel=...)."""
    tgt, twk = cfg["target"], cfg["time_col"]
    p = panel.copy()
    p[tgt] = pd.to_numeric(p[tgt], errors="coerce").fillna(0.0)
    if "exclude_forecast" in p.columns:                       # score grain = non-excluded keys
        p = p[pd.to_numeric(p["exclude_forecast"], errors="coerce") == 0]
    p["cs_gtin"] = p["key"].astype(str).str.rsplit("_", n=1).str[0]
    aggm = {}
    for c in p.columns:
        if c in ("cs_gtin", twk, "key"):
            continue
        lc = c.lower()
        if c == tgt or any(k in lc for k in ("sales", "cases", "units", "quantity", "volume", "smooth", "gbp")):
            aggm[c] = "sum"
        elif any(k in lc for k in ("rate", "price", "ratio", "dist", "wthr", "temp")):
            aggm[c] = "mean"
        elif pd.api.types.is_numeric_dtype(p[c]):
            aggm[c] = "max"                                   # holiday/season/promo flags, indicators
        else:
            aggm[c] = "first"                                 # category/brand/APG/hierarchy
    g = p.groupby(["cs_gtin", twk], as_index=False).agg(aggm).rename(columns={"cs_gtin": "key"})
    return g


__all__ = ["load_config", "add_weeks", "read_input_panel", "forecast_origin",
           "forecast_origin_stacked", "aggregate_to_gtin"]
