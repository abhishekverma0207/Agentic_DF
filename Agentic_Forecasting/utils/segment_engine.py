"""
Production segment-model engine — the validated `segment(category × lego_segment) ×
category × LEGO` stacked ensemble.

ISOLATED from ``utils.uk_engine`` (which UK's LIVE path uses) so adding the segment
layer carries ZERO UK-regression risk: UK keeps running through uk_engine unchanged;
DE routes here when ``market_io.segment_col`` is configured.

Validated out-of-sample: beats LEGO on 9/10 DE categories (acc + asymmetric-bias),
no regressions; the head is LEGO's per-key ceiling so the blend correctly defers
to LEGO there. See ``memory/de-segment-model.md``.

Entry points (used by both FIT = utils.weight_fit and LIVE = utils.diq_runner):
  * ``compute_segment_components(cfg, full_df, ...)`` -> per-row [key, year_week,
    horizon, s, c, rf, category, segment, cs_gtin]. Trains category model ``c`` and
    per-(category × segment) model ``s`` in a worker-capped process pool, reads LEGO
    ``rf`` = ``lego_pred_col``.
  * ``fit_segment_weights(components_with_actuals, ...)`` -> per-(category × segment)
    weights [w_s, w_c, w_rf] with hierarchical shrinkage toward the category blend +
    LEGO guardrail.  (FIT)
  * ``blend_with_weights(components, weights)`` -> ``predicted`` per row.  (LIVE)
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Knobs (overridable via env, matching the validated research defaults)
SEG_MIN_ROWS = int(os.environ.get("SEG_MIN_ROWS", "400"))       # min rows to train a (cat×seg) model
CAT_MIN_ROWS = 50
MIN_SEG_VOLUME = float(os.environ.get("SEG_MIN_VOLUME", "1000"))  # min realized volume to deploy a stack
SHRINK_K = float(os.environ.get("SEG_SHRINK_K", "5000"))        # vol at which a segment is 50% shrunk to its category
RECENCY_HL = 39.0
RAW_CAT, RAW_SEG = "_seg_cat_raw", "_seg_seg_raw"


# --------------------------------------------------------------------------- #
# Worker — train one group (category OR category×segment) for one horizon.
# Module-level for spawn; mirrors uk_engine._train_and_predict_one + a seg filter.
# --------------------------------------------------------------------------- #
def _worker_init(project_root: str) -> None:
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _train_one(args: tuple) -> dict:
    (kind, h, cat, seg, panel_path, train_end, key_col, week_col, target_col,
     feat_cols, cat_cols, lgb_params, n_boost) = args
    import lightgbm as lgb
    from utils.uk_forecast import add_weeks
    cols = list({key_col, week_col, target_col, RAW_CAT, RAW_SEG, "_wi", *feat_cols})
    df = pd.read_parquet(panel_path, columns=[c for c in cols if c is not None])
    if kind == "category":
        df = df[df[RAW_CAT] == cat]
        min_rows = CAT_MIN_ROWS
    else:
        df = df[(df[RAW_CAT] == cat) & (df[RAW_SEG] == seg)]
        min_rows = SEG_MIN_ROWS
    empty = {"kind": kind, "cat": cat, "seg": seg, "h": h, "pred": None}
    if len(df) < min_rows:
        return empty
    tw = {w: add_weeks(w, h) for w in df[week_col].unique()}
    left = df[[key_col, week_col, "_wi", *feat_cols]].copy()
    left["_tw"] = left[week_col].map(tw)
    right = df[[key_col, week_col, target_col]].rename(columns={week_col: "_tw", target_col: "_y"})
    pairs = left.merge(right, on=[key_col, "_tw"], how="inner")
    tr = pairs[pairs["_tw"] <= train_end]
    pr = pairs[pairs[week_col] == train_end]
    if tr.empty or pr.empty:
        return empty
    labels = np.clip(tr["_y"].values, 0, None)
    if labels.sum() == 0:
        return empty                      # tweedie requires non-zero label sum
    te = tr["_wi"].max()
    w = np.power(0.5, (te - tr["_wi"]).clip(lower=0) / RECENCY_HL)
    dtrain = lgb.Dataset(tr[feat_cols], label=np.clip(tr["_y"].values, 0, None), weight=w.values,
                         categorical_feature=cat_cols, free_raw_data=False)
    model = lgb.train(lgb_params, dtrain, num_boost_round=n_boost)
    out = pr[[key_col, "_tw"]].rename(columns={"_tw": week_col}).copy()
    out["value"] = np.clip(model.predict(pr[feat_cols]), 0, None)
    out["horizon"] = h
    return {"kind": kind, "cat": cat, "seg": seg, "h": h, "pred": out}


# --------------------------------------------------------------------------- #
# Component builder (FIT + LIVE)
# --------------------------------------------------------------------------- #
def compute_segment_components(
    cfg, full_df: pd.DataFrame, *, snapshot_week: str, route_categories: Sequence[str],
    segment_col: str, lego_source: str, config_path: str, lego_pred_col: str,
    horizons: Optional[int] = None, workers: Optional[int] = None,
) -> pd.DataFrame:
    """Build the DT⋈FS feature panel, train category ``c`` + per-(category×segment) ``s``
    models, read LEGO ``rf``, and return per-row components for the forward weeks."""
    from utils.uk_forecast import (
        load_config, add_weeks, add_backward_features, _encode_categoricals, _future_cols, _lgb_params,
    )
    from utils.uk_engine import _read_lego, DEFAULT_LEGO_KEY_COL, DEFAULT_LEGO_WEEK_COL

    model_cfg = load_config(_resolve(config_path), overrides={
        "input_source": "dbx", "collapse_promo": False, "holiday_distance_features": False,
        "rich_features": False, "per_category": False, "bias_calibration": False})
    cat_col = model_cfg["cat_col"]
    key_col, week_col, target = model_cfg["key_col"], "year_week", model_cfg["target"]
    H = int(horizons or getattr(cfg, "forecast_horizon", model_cfg.get("horizons", 13)))

    df = full_df.copy()
    if route_categories:
        df = df[df[cat_col].astype(str).isin([str(c) for c in route_categories])]
    df = df.sort_values([key_col, week_col], kind="mergesort").reset_index(drop=True)
    df[week_col] = df[week_col].astype(str).str.replace("-", "", regex=False)
    df[key_col] = df[key_col].astype(str)
    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
    df[RAW_CAT] = df[cat_col].astype(str)
    df[RAW_SEG] = df[segment_col].astype(str)

    model_cfg = {**model_cfg, "cat_col": cat_col}
    back = add_backward_features(df, model_cfg)
    catc = _encode_categoricals(df, model_cfg)
    futc = _future_cols(df, model_cfg)
    feat = list(dict.fromkeys(back + catc + futc))
    weeks = sorted(df[week_col].unique())
    df["_wi"] = df[week_col].map({w: i for i, w in enumerate(weeks)}).astype("int32")
    train_end = add_weeks(snapshot_week, -1)

    panel_path = str(Path(tempfile.gettempdir()) / f"segeng_panel_{snapshot_week}.parquet")
    keep = list(dict.fromkeys([key_col, week_col, "_wi", target, RAW_CAT, RAW_SEG, *feat]))
    df[keep].to_parquet(panel_path, index=False)

    cats = sorted(df[RAW_CAT].unique())
    seg_by_cat = {c: sorted(df[df[RAW_CAT] == c][RAW_SEG].unique()) for c in cats}
    keymap = df[[key_col, RAW_CAT, RAW_SEG]].drop_duplicates(key_col)

    lgb = _lgb_params(model_cfg)
    lgb["num_threads"] = int(os.environ.get("UK_ENGINE_THREADS", "4"))
    n_boost = int(model_cfg.get("n_estimators", 700))
    horizons_list = list(range(1, H + 1))
    common = (panel_path, train_end, key_col, week_col, target, feat, catc)
    tasks = []
    for h in horizons_list:
        for c in cats:
            tasks.append(("category", h, c, None, *common, dict(lgb), n_boost))
            for s in seg_by_cat[c]:
                tasks.append(("segment", h, c, s, *common, dict(lgb), n_boost))

    threads = lgb["num_threads"]
    default_workers = max(1, (os.cpu_count() or 8) // threads - 1)
    n_workers = int(workers or os.environ.get("UK_ENGINE_WORKERS", default_workers))
    logger.info("[segment_engine] %s: %d tasks (%d cats, %d cat×seg groups), %d workers × %d threads",
                snapshot_week, len(tasks), len(cats),
                sum(len(v) for v in seg_by_cat.values()), n_workers, threads)

    project_root = str(Path(__file__).resolve().parents[1])
    cat_pred, seg_pred = {}, {}

    # UK_ENGINE_EXECUTOR: "process" (default, best on Linux/Databricks),
    #                     "thread"  (safe on macOS, still parallel).
    executor_mode = os.environ.get("UK_ENGINE_EXECUTOR", "process").lower()

    if executor_mode == "thread":
        from concurrent.futures import ThreadPoolExecutor as TPool
        logger.info("[segment_engine] using ThreadPoolExecutor (%d workers)", n_workers)
        _worker_init(project_root)
        pool_cls = lambda: TPool(max_workers=n_workers)
    else:
        ctx = mp.get_context("spawn")
        logger.info("[segment_engine] using ProcessPoolExecutor (%d workers)", n_workers)
        pool_cls = lambda: ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                               initializer=_worker_init, initargs=(project_root,))

    with pool_cls() as pool:
        for fut in as_completed([pool.submit(_train_one, t) for t in tasks]):
            r = fut.result()
            if r["pred"] is None:
                continue
            (cat_pred if r["kind"] == "category" else seg_pred)[
                (r["cat"], r["h"]) if r["kind"] == "category" else (r["cat"], r["seg"], r["h"])] = r["pred"]

    c_df = (pd.concat([d for d in cat_pred.values()], ignore_index=True).rename(columns={"value": "c"})
            if cat_pred else pd.DataFrame(columns=[key_col, week_col, "horizon", "c"]))
    s_df = (pd.concat([d for d in seg_pred.values()], ignore_index=True).rename(columns={"value": "s"})
            if seg_pred else pd.DataFrame(columns=[key_col, week_col, "horizon", "s"]))
    comp = c_df[[key_col, week_col, "horizon", "c"]].merge(
        s_df[[key_col, week_col, "horizon", "s"]], on=[key_col, week_col, "horizon"], how="left")

    lego_df = _read_lego(lego_source, pred_col=lego_pred_col)
    lego_df = lego_df.rename(columns={DEFAULT_LEGO_KEY_COL: key_col, DEFAULT_LEGO_WEEK_COL: week_col,
                                      lego_pred_col: "rf"})
    lego_df[key_col] = lego_df[key_col].astype(str)
    lego_df[week_col] = lego_df[week_col].astype(str).str.replace("-", "", regex=False)
    comp = comp.merge(lego_df[[key_col, week_col, "rf"]].drop_duplicates([key_col, week_col]),
                      on=[key_col, week_col], how="left")
    comp = comp.merge(keymap, on=key_col, how="left")
    comp["category"] = comp[RAW_CAT]
    comp["segment"] = comp[RAW_SEG]
    comp["cs_gtin"] = comp[key_col].str.rsplit("_", n=1).str[0]
    comp["s"] = comp["s"].fillna(comp["c"])     # tiny/absent segment -> category model
    comp["rf"] = comp["rf"].fillna(0.0)
    return comp.rename(columns={key_col: "key"})[
        ["key", week_col, "horizon", "s", "c", "rf", "category", "segment", "cs_gtin"]]


def _resolve(p: str) -> str:
    try:
        from utils.uk_engine import _resolve_repo_path
        return _resolve_repo_path(p)
    except Exception:
        return p


# --------------------------------------------------------------------------- #
# FIT — per-(category × segment) weights with hierarchical shrinkage + guardrail
# --------------------------------------------------------------------------- #
def fit_segment_weights(df: pd.DataFrame, *, grain: str = "gtin", objective: str = "wape_asym",
                        bias_tolerance: float = 0.02, guardrail: bool = True, acc_margin: float = 0.0,
                        shrink: bool = True, week_col: str = "year_week",
                        calibrate: bool = False) -> Dict[tuple, dict]:
    """``df`` has [key, year_week, s, c, rf, actual, category, segment, cs_gtin]. Returns
    {(category, segment): {weights:[w_s,w_c,w_rf], use, acc_*, bias_*}}."""
    if calibrate:
        # "E" path: bias-CALIBRATE the blend (fold a global scale into the weights), then run the
        # guardrail on the CALIBRATED blend per category. Validated OOS (202616): +5.8pp acc vs LEGO
        # at 9/10 category wins & clean bias — vs +1.4pp for the plain guardrail on the raw blend.
        return _fit_segment_calibrated(df, grain=grain, objective=objective,
                                       bias_tolerance=bias_tolerance, acc_margin=acc_margin, week_col=week_col)
    from utils.weight_fit import fit_per_category_weights
    # Volume floor + shrink read at CALL time so the env knobs (SEG_MIN_VOLUME / SEG_SHRINK_K)
    # apply without a Python restart. Defaults preserve the validated behavior (1000 / 5000).
    min_seg_volume = float(os.environ.get("SEG_MIN_VOLUME", MIN_SEG_VOLUME))
    shrink_k = float(os.environ.get("SEG_SHRINK_K", SHRINK_K))
    d = df.assign(g=df["s"], l=df["c"],
                  _grp=df["category"].astype(str) + " | " + df["segment"].astype(str))
    kw = dict(grain=grain, objective=objective, bias_tolerance=bias_tolerance, guardrail=guardrail,
              acc_margin=acc_margin, lego_col="rf", gtin_col="cs_gtin", week_col=week_col)
    seg = fit_per_category_weights(d, cat_col="_grp", **kw)
    cat = fit_per_category_weights(d, cat_col="category", **kw)
    vol = d.groupby("_grp")["actual"].sum()
    g2c = d.drop_duplicates("_grp").set_index("_grp")["category"].to_dict()
    out = {}
    for k, v in seg.items():
        cat_name = g2c.get(k)
        seg_name = k.split(" | ", 1)[-1]
        gk = (cat_name, seg_name)
        cw = cat.get(cat_name, {"weights": [0.0, 0.0, 1.0], "use": "LEGO"})
        if float(vol.get(k, 0.0)) < min_seg_volume:
            out[gk] = {**v, "weights": [0.0, 0.0, 1.0], "use": "LEGO(lowvol)"}
            continue
        if shrink:
            _v = float(vol.get(k, 0.0))
            lam = 1.0 if shrink_k <= 0 else _v / (_v + shrink_k)   # shrink_k=0 → raw per-segment weights
            w = [lam * v["weights"][i] + (1 - lam) * cw["weights"][i] for i in range(3)]
            use = "STACK" if (v["use"] == "STACK" or cw["use"] == "STACK") else "LEGO"
            out[gk] = {**v, "weights": w if use == "STACK" else [0.0, 0.0, 1.0], "use": use}
        else:
            out[gk] = v
    return out


def _gtin_acc_bias(sub, pred_col, week_col):
    """1−WAPE accuracy and bias for `pred_col` vs actual, aggregated to GTIN×week grain."""
    g = sub.groupby(["cs_gtin", week_col]).agg(p=(pred_col, "sum"), a=("actual", "sum")).reset_index()
    p, a = g["p"].to_numpy(float), g["a"].to_numpy(float)
    tot = max(a.sum(), 1e-9)
    return 1.0 - np.abs(p - a).sum() / tot, (p - a).sum() / tot


def _fit_segment_calibrated(df, *, grain, objective, bias_tolerance, acc_margin, week_col):
    """E — global bias calibration folded into the weights, then a per-category guardrail on the
    CALIBRATED blend. Steps: raw (un-guardrailed) per-(cat×seg) fit + shrink toward category →
    scale every weight by k = Σactual / Σblend (clipped) so the blend's level matches actuals →
    keep a category's calibrated weights only if they beat LEGO on acc+bias here, else pure LEGO.
    k is folded into the weights, so the LIVE path is unchanged. SEG_CAL_LO/HI bound the scale."""
    from utils.weight_fit import fit_per_category_weights
    shrink_k = float(os.environ.get("SEG_SHRINK_K", SHRINK_K))
    cal_lo = float(os.environ.get("SEG_CAL_LO", "0.5"))
    cal_hi = float(os.environ.get("SEG_CAL_HI", "2.0"))
    d = df.assign(g=df["s"], l=df["c"],
                  _grp=df["category"].astype(str) + " | " + df["segment"].astype(str))
    kw = dict(grain=grain, objective=objective, bias_tolerance=bias_tolerance, guardrail=False,
              acc_margin=0.0, lego_col="rf", gtin_col="cs_gtin", week_col=week_col)
    seg = fit_per_category_weights(d, cat_col="_grp", **kw)              # raw segment weights
    cat = fit_per_category_weights(d, cat_col="category", **kw)          # raw category weights (shrink prior)
    vol = d.groupby("_grp")["actual"].sum()
    g2c = d.drop_duplicates("_grp").set_index("_grp")["category"].to_dict()

    raw = {}
    for k, v in seg.items():
        cw = cat.get(g2c.get(k), {"weights": [0.0, 0.0, 1.0]})
        _v = float(vol.get(k, 0.0))
        lam = 1.0 if shrink_k <= 0 else _v / (_v + shrink_k)
        raw[k] = [lam * v["weights"][i] + (1.0 - lam) * cw["weights"][i] for i in range(3)]

    # global calibration scale on the fit data: k = Σactual / Σblend
    W = np.array([raw.get(g, [0.0, 0.0, 1.0]) for g in d["_grp"]], dtype=float)
    blend = np.clip(W[:, 0] * d["s"].to_numpy(float) + W[:, 1] * d["c"].to_numpy(float)
                    + W[:, 2] * d["rf"].to_numpy(float), 0, None)
    kcal = float(np.clip(d["actual"].sum() / max(blend.sum(), 1e-9), cal_lo, cal_hi))
    dd = d.assign(_blend=blend * kcal)

    catstats = {}
    for c, sub in dd.groupby("category"):
        aL, bL = _gtin_acc_bias(sub, "rf", week_col)
        aC, bC = _gtin_acc_bias(sub, "_blend", week_col)
        catstats[c] = (aL, bL, aC, bC)

    out = {}
    for k in seg:
        c = g2c.get(k)
        gk = (c, k.split(" | ", 1)[-1])
        aL, bL, aC, bC = catstats.get(c, (0.0, 0.0, 0.0, 0.0))
        keep = (aC >= aL - 1e-3 + acc_margin) and (min(bC, 0.0) >= min(bL, 0.0) - bias_tolerance)
        out[gk] = {"weights": [round(kcal * raw[k][i], 4) for i in range(3)] if keep else [0.0, 0.0, 1.0],
                   "use": "CAL" if keep else "LEGO",
                   "acc_lego": round(aL, 4), "acc_stack": round(aC, 4),
                   "bias_lego": round(bL, 4), "bias_stack": round(bC, 4)}
    return out


def load_segment_weights(path_or_dict):
    """Load the per-(category × segment) weights YAML written by FIT.
    Returns ({(category, segment): [w_s, w_c, w_rf]}, default=[w_s,w_c,w_rf])."""
    import yaml
    d = path_or_dict if isinstance(path_or_dict, dict) else yaml.safe_load(open(path_or_dict))
    out: Dict[tuple, list] = {}
    for cat, segs in (d.get("segment_weights") or {}).items():
        for seg, w in segs.items():
            out[(str(cat), str(seg))] = [float(x) for x in w]
    return out, [float(x) for x in d.get("default_weights", [0.0, 0.0, 1.0])]


# --------------------------------------------------------------------------- #
# LIVE — apply per-segment weights
# --------------------------------------------------------------------------- #
def blend_with_weights(comp: pd.DataFrame, weights: Dict[tuple, list], *,
                       default: Optional[list] = None) -> np.ndarray:
    default = default or [0.0, 0.0, 1.0]
    grp = list(zip(comp["category"].astype(str), comp["segment"].astype(str)))
    W = np.array([weights.get(k, weights.get((k[0], "_default"), default)) for k in grp], dtype=float)
    return np.clip(W[:, 0] * comp["s"].values + W[:, 1] * comp["c"].values + W[:, 2] * comp["rf"].values, 0, None)
