"""
Generic, market-agnostic FIT mode for the stacked DIQ ensemble.

Estimates per-category blend weights ``[w_global, w_local, w_lego]`` from several
rolling-origin snapshots, then writes them to ``config/{market}_ensemble_weights.yaml``
in the schema ``utils.uk_engine._load_weights`` reads. The LIVE path
(``utils.diq_runner.run_diq_forecast`` -> ``run_uk_engine``) then applies the frozen
weights unchanged.

FIT is two explicit stages (matching the Databricks notebook flow — "run our new
DIQ, save them, then use the DIQ output + the LEGO benchmark to build the fit"):
  * ``save_components(market, snapshot, out_dir=...)`` — STAGE A. Runs the new DIQ
    (``uk_engine.compute_components``: global ``g`` + per-cat ``l`` GBMs + LEGO ``rf``)
    for ONE snapshot and persists the component frame to parquet. The expensive step.
  * ``fit_from_components(market, components_dir=..., actuals_source=...)`` — STAGE B.
    Reads the saved components + realized forward actuals, joins, pools, fits, writes
    ``{market}_ensemble_weights.yaml``. Cheap → re-runnable on different objectives.
  * ``fit_per_category_weights`` — the pure optimiser (no I/O) underneath Stage B.
  * ``fit_weights`` — convenience one-shot (Stage A for every snapshot, then Stage B);
    used by ``run_stack.py --mode fit`` and local validation.

Everything market-specific (LEGO column, paths, snapshots, objective) comes from
``config_{market}_base_local.yaml`` — UK vs DE is a pure ``market=`` switch.
"""
from __future__ import annotations

import glob
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics (pooled ratio-of-sums, same convention as utils.lego_eval / the UK work)
# ---------------------------------------------------------------------------
def _acc(f, a) -> float:
    f = np.asarray(f, float); a = np.asarray(a, float)
    return 1.0 - np.abs(f - a).sum() / max(a.sum(), 1e-9)


def _bias(f, a) -> float:
    f = np.asarray(f, float); a = np.asarray(a, float)
    return (f - a).sum() / max(a.sum(), 1e-9)


# ---------------------------------------------------------------------------
# Pure optimiser — ported from /tmp/uk_asym.py (validated asymmetric fit)
# ---------------------------------------------------------------------------
def fit_per_category_weights(
    df: pd.DataFrame,
    *,
    categories: Optional[Sequence[str]] = None,
    grain: str = "gtin",
    objective: str = "wape_asym",
    bias_tolerance: float = 0.02,
    guardrail: bool = True,
    acc_margin: float = 0.0,
    lego_col: str = "rf",
    cat_col: str = "category",
    gtin_col: str = "cs_gtin",
    week_col: str = "year_week",
) -> Dict[str, dict]:
    """Fit ``[w_g, w_l, w_lego]`` per category on ``df`` (one row per
    key/week/horizon with columns g, l, <lego_col>, actual, category, gtin).

    objective:
      * ``wape_asym`` — minimise GTIN-WAPE with a one-sided penalty on UNDER-forecast
        beyond ``bias_tolerance`` vs LEGO (over-forecast is free). The rule we validated.
      * ``wape``      — minimise GTIN-WAPE only.
    guardrail: deploy the fitted stack for a category only if it beats LEGO on BOTH
      accuracy and the asymmetric-bias rule (else fall back to pure LEGO weights).

    Returns ``{category: {"weights": [w_g, w_l, w_lego], "use": "STACK"|"LEGO",
                          "acc_lego","acc_stack","bias_lego","bias_stack"}}``.
    """
    INP = ["g", "l", lego_col]
    d = df.copy()
    for c in INP + ["actual"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    cats = list(categories) if categories else sorted(d[cat_col].dropna().unique())
    group_keys = ([gtin_col, week_col] if grain == "gtin" else None)

    def _aggregate(sub: pd.DataFrame):
        if group_keys:
            a = sub.groupby(group_keys)[INP + ["actual"]].sum()
            return a[INP].values, a["actual"].values
        return sub[INP].values, sub["actual"].values

    def _fit(X, A):
        tot = max(A.sum(), 1e-9)
        lego_bias = (X[:, 2] - A).sum() / tot
        floor = min(lego_bias, 0.0) - bias_tolerance
        def obj(w):
            pred = X @ w
            wape = np.abs(pred - A).sum() / tot
            if objective == "wape":
                return wape
            bM = (pred - A).sum() / tot
            under = max(0.0, floor - min(bM, 0.0))   # only penalise under-forecast worse than LEGO+tol
            return wape + 200.0 * under ** 2
        starts = [[0, 0, 1], [0, 0.5, 0.6], [0.2, 0.5, 0.4], [0.3, 0.3, 0.4]]
        best = min(
            (minimize(obj, s, method="Nelder-Mead", bounds=[(0, 3)] * 3,
                      options={"maxiter": 1000, "xatol": 1e-3, "fatol": 1e-8}) for s in starts),
            key=lambda r: r.fun)
        return [round(float(v), 4) for v in best.x]

    out: Dict[str, dict] = {}
    for cat in cats:
        sub = d[d[cat_col] == cat]
        if sub.empty:
            continue
        X, A = _aggregate(sub)
        w = _fit(X, A)
        stack = X @ np.array(w)
        lego = X[:, 2]
        aL, aS = _acc(lego, A), _acc(stack, A)
        bL, bS = _bias(lego, A), _bias(stack, A)
        # acc_margin (>=0) hardens the guardrail: a stack must beat LEGO by this much on the
        # fit data before it deploys — insures against out-of-sample flips (segment overfit).
        beats = (aS >= aL - 1e-3 + acc_margin) and (min(bS, 0.0) >= min(bL, 0.0) - bias_tolerance)
        use = "STACK" if (not guardrail or beats) else "LEGO"
        out[cat] = {
            "weights": w if use == "STACK" else [0.0, 0.0, 1.0],
            "use": use,
            "acc_lego": round(aL, 4), "acc_stack": round(aS, 4),
            "bias_lego": round(bL, 4), "bias_stack": round(bS, 4),
        }
    return out


# ---------------------------------------------------------------------------
# Generic parquet IO — NATIVE when the path is reachable (on a Databricks cluster
# the /Volumes/ paths are FUSE-mounted; local datacache is a plain dir), else the
# Databricks SDK (off-cluster local dev). The SAME code runs in the notebook and
# in local validation — only the reach changes.
# ---------------------------------------------------------------------------
def _read_delta_any(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a Delta table dir to pandas. Prefer Spark (present on Databricks, so
    the transaction log is honoured); fall back to deltalake for local dev. A
    plain pyarrow read would include tombstoned/rewritten files and return the
    wrong rows, so we never fall back to that for a Delta path."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.read.format("delta").load(str(path))
        if columns:
            sdf = sdf.select(*columns)
        return sdf.toPandas()
    except Exception:
        from deltalake import DeltaTable
        return DeltaTable(str(path)).to_pandas(columns=list(columns) if columns else None)


def read_parquet_any(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    p = str(path)
    remote = p.startswith("/Volumes/") or p.startswith("dbfs:")
    if remote and not os.path.exists(p):
        from utils import uc_io                      # off-cluster: Volume not mounted -> SDK pull
        return uc_io.read_parquet(p, columns=columns, max_workers=16)
    if os.path.isdir(p):                             # mounted Volume dir, or local parquet dir
        if os.path.isdir(os.path.join(p, "_delta_log")):   # Delta table -> honour the log
            return _read_delta_any(p, columns=columns)
        import pyarrow.parquet as pq
        return pq.read_table(p, columns=columns).to_pandas()
    return pd.read_parquet(p, columns=columns)       # single file


def write_parquet_any(df: pd.DataFrame, path: str) -> str:
    """Write one parquet file. Native on-cluster (/Volumes/ FUSE) and local; SDK
    upload off-cluster only if utils.uc_io exposes a writer."""
    p = str(path)
    parent = os.path.dirname(p)
    if (p.startswith("/Volumes/") or p.startswith("dbfs:")) and not os.path.isdir(parent):
        try:
            from utils import uc_io
            if hasattr(uc_io, "write_parquet"):
                uc_io.write_parquet(df, p)
                return p
        except Exception:
            logger.warning("[weight_fit] uc_io.write_parquet unavailable; trying native write to %s", p)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _load_cfg(market: str, config_root: str) -> dict:
    from utils.config_loader import load_market_config
    return load_market_config(market, config_root)   # prefers config_{market}_base_local.py, else .yaml


def _wf_params(market: str, config_root: str):
    cfg = _load_cfg(market, config_root)
    return cfg, cfg.get("market_io", {}), cfg.get("weight_fit", {})


# ===========================================================================
# STAGE A — run the new DIQ for ONE snapshot and SAVE its components.
# This is "run our new DIQ, save them". The expensive step (trains global +
# per-category GBMs); persisting it lets Stage B be re-run on different
# objectives/tolerances without re-training.
# ===========================================================================
def save_components(
    market: str,
    snapshot: str,
    *,
    out_dir: str,
    config_root: str = "config",
    parent_dir_template: Optional[str] = None,
    panel_path_template: Optional[str] = None,
    lego_path_template: Optional[str] = None,
    in_scope_categories: Optional[Sequence[str]] = None,
    horizons: Optional[int] = None,
) -> dict:
    """Run ``uk_engine.compute_components`` (global ``g`` + per-category ``l`` GBMs,
    plus the LEGO benchmark ``rf`` = ``lego_pred_col``) for one snapshot and write
    ``[key, year_week, horizon, g, l, rf, predicted, category, cs_gtin, snapshot]``
    to ``{out_dir}/components_{snapshot}.parquet``. Returns a small status dict.

    Panel source (pick one; ``parent_dir_template`` preferred for production fidelity):
      * ``parent_dir_template`` — ``{snapshot}``-templated dir holding ``{cats_subdir}/``
        and ``{fs_subdir}/``. The panel is built via ``diq_runner._load_and_join_inputs``
        — the SAME DT⋈FS join the LIVE runner uses, so FIT ``g``/``l`` match LIVE exactly.
      * ``panel_path_template`` — a single already-joined parquet dir (e.g. a local cache)."""
    from utils import uk_engine

    cfg, mio, wf = _wf_params(market, config_root)
    horizons = int(horizons or wf.get("horizons", 13))
    lego_pred_col = mio.get("lego_pred_col", "prediction_rf")
    cat_col = mio.get("category_col", "category_name")
    parent_tmpl = parent_dir_template or wf.get("parent_dir_template")
    panel_tmpl = panel_path_template or wf.get("panel_path_template")
    lego_tmpl = lego_path_template or wf["lego_path_template"]
    in_scope = in_scope_categories or wf.get("in_scope_categories")
    config_path = str(Path(config_root) / f"config_{market}_base_local.yaml")

    # fit-time weights file: force g+l training for ALL cats (no lego-only short-circuit)
    fit_w = Path(tempfile.gettempdir()) / f"_fitweights_{market}.yaml"
    yaml.safe_dump({"weights": {}, "default_weights": {"global": 0.34, "local": 0.33, "lego": 0.33},
                    "short_circuit_lego_only": False}, open(fit_w, "w"))

    if parent_tmpl:
        # production-fidelity panel: DT ⋈ FS exactly like utils.diq_runner LIVE path
        from utils.diq_runner import _load_and_join_inputs
        pdir = Path(parent_tmpl.format(snapshot=snapshot))
        panel = _load_and_join_inputs(
            pdir, category_list=list(in_scope) if in_scope else None,
            category_col=cat_col, cats_subdir=mio.get("cats_subdir", "DT"),
            fs_subdir=mio.get("fs_subdir", "FS"))
    elif panel_tmpl:
        panel = read_parquet_any(panel_tmpl.format(snapshot=snapshot))
    else:
        raise ValueError("save_components: set parent_dir_template (DT⋈FS, preferred) "
                         "or panel_path_template (pre-joined panel) in config or kwargs.")
    panel["year_week"] = panel["year_week"].astype(str)
    panel["key"] = panel["key"].astype(str)
    if in_scope and cat_col in panel.columns:
        panel = panel[panel[cat_col].isin(in_scope)].copy()
    route = sorted(panel[cat_col].astype(str).unique())
    eng_cfg = SimpleNamespace(forecast_horizon=horizons,
                              artifact_base_path=str(Path(tempfile.gettempdir()) / f"fit_{market}_{snapshot}"))
    segment_col = mio.get("segment_col")
    if segment_col:
        # segment mode: trains category(c) + per-(category×segment) models (s) → [s, c, rf]
        from utils import segment_engine
        comp = segment_engine.compute_segment_components(
            eng_cfg, panel, snapshot_week=snapshot, route_categories=route,
            segment_col=segment_col, lego_source=lego_tmpl.format(snapshot=snapshot),
            config_path=config_path, lego_pred_col=lego_pred_col, horizons=horizons)
    else:
        comp = uk_engine.compute_components(
            eng_cfg, panel, snapshot_week=snapshot, route_categories=route,
            weights_path=str(fit_w), lego_source=lego_tmpl.format(snapshot=snapshot),
            config_path=config_path, lego_pred_col=lego_pred_col)
    comp["key"] = comp["key"].astype(str)
    comp["year_week"] = comp["year_week"].astype(str).str.replace("-", "", regex=False)
    comp["cs_gtin"] = comp["key"].str.rsplit("_", n=1).str[0]
    comp["snapshot"] = str(snapshot)
    out_path = f"{str(out_dir).rstrip('/')}/components_{snapshot}.parquet"
    write_parquet_any(comp, out_path)
    cats = sorted(comp["category"].astype(str).unique()) if "category" in comp.columns else route
    logger.info("[weight_fit] STAGE A %s: %d rows -> %s (%d cats)", snapshot, len(comp), out_path, len(cats))
    return {"snapshot": str(snapshot), "path": out_path, "rows": int(len(comp)), "categories": cats}


# ===========================================================================
# STAGE B — fit per-category weights from the SAVED DIQ components + the LEGO
# benchmark (rf, already inside the saved frame) + realized forward actuals.
# This is "use the DIQ output and the prediction_xgb to then build the fit".
# ===========================================================================
def fit_from_components(
    market: str,
    *,
    components_dir: Optional[str] = None,
    component_paths: Optional[Sequence[str]] = None,
    actuals_source: Optional[str] = None,
    config_root: str = "config",
    out_path: Optional[str] = None,
    grain: Optional[str] = None,
    objective: Optional[str] = None,
    bias_tolerance: Optional[float] = None,
    guardrail: Optional[bool] = None,
    return_detail: bool = False,
):
    """Read the saved Stage-A component frames + realized forward actuals, join on
    ``(key, year_week)``, pool across snapshots, fit per-category ``[w_g, w_l, w_lego]``,
    and write ``{market}_ensemble_weights.yaml``. Returns the written path (or, when
    ``return_detail`` is set, ``{"path": ..., "fitted": {cat: {...}}}`` for display)."""
    cfg, mio, wf = _wf_params(market, config_root)
    grain = grain or wf.get("grain", "gtin")
    objective = objective or wf.get("objective", "wape_asym")
    bias_tol = float(bias_tolerance if bias_tolerance is not None else wf.get("bias_tolerance", 0.02))
    guardrail = bool(guardrail if guardrail is not None else wf.get("guardrail", True))
    actuals_source = actuals_source or wf["actuals_source"]

    # resolve the saved component parts
    paths = list(component_paths or [])
    if not paths and components_dir:
        cd = str(components_dir)
        if (cd.startswith("/Volumes/") or cd.startswith("dbfs:")) and not os.path.exists(cd):
            from utils import uc_io
            paths = sorted(e.path for e in uc_io.ls(cd)
                           if e.name.startswith("components_") and e.name.endswith(".parquet"))
        else:
            paths = sorted(glob.glob(os.path.join(cd, "components_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no component parquet found (dir={components_dir}, paths={component_paths})")

    comps = []
    for p in paths:
        c = read_parquet_any(p)
        c["key"] = c["key"].astype(str)
        c["year_week"] = c["year_week"].astype(str).str.replace("-", "", regex=False)
        if "cs_gtin" not in c.columns:
            c["cs_gtin"] = c["key"].str.rsplit("_", n=1).str[0]
        comps.append(c)
    comp = pd.concat(comps, ignore_index=True)

    # realized forward actuals (read only key/week/target to stay light)
    key_col = cfg.get("key_col", "key"); time_col = cfg.get("time_col", "year_week")
    tgt = cfg.get("target", "Actuals")
    try:
        acts = read_parquet_any(actuals_source, columns=[key_col, time_col, tgt])
    except Exception:
        acts = read_parquet_any(actuals_source)
    acts = acts.rename(columns={key_col: "key", time_col: "year_week", tgt: "actual"})
    acts = acts.rename(columns={c: "actual" for c in acts.columns if c.lower() in ("actuals", "actual_sales", "actual")})
    acts["key"] = acts["key"].astype(str)
    acts["year_week"] = acts["year_week"].astype(str).str.replace("-", "", regex=False)
    acts = acts[["key", "year_week", "actual"]].groupby(["key", "year_week"], as_index=False)["actual"].sum()

    df = comp.merge(acts, on=["key", "year_week"], how="inner")
    nsnap = comp["snapshot"].nunique() if "snapshot" in comp.columns else 1
    logger.info("[weight_fit] STAGE B: %d component rows, %d joined to actuals across %d snapshot(s)",
                len(comp), len(df), nsnap)

    if mio.get("segment_col") and "segment" in df.columns:
        # segment mode: per-(category × segment) weights with hierarchical shrinkage + guardrail
        from utils import segment_engine
        fitted = segment_engine.fit_segment_weights(
            df, grain=grain, objective=objective, bias_tolerance=bias_tol, guardrail=guardrail,
            acc_margin=float(wf.get("guardrail_margin", 0.0)),
            calibrate=bool(wf.get("calibrate", False)))
        path = _write_segment_weights_yaml(fitted, market, config_root, out_path)
    else:
        fitted = fit_per_category_weights(
            df, grain=grain, objective=objective, bias_tolerance=bias_tol,
            guardrail=guardrail, lego_col="rf", cat_col="category")
        path = _write_weights_yaml(fitted, market, config_root, out_path)
    return {"path": path, "fitted": fitted} if return_detail else path


def _write_segment_weights_yaml(fitted: Dict[tuple, dict], market: str, config_root: str,
                                out_path: Optional[str]) -> str:
    """Serialize per-(category × segment) weights [w_s, w_c, w_rf] for the LIVE blend.
    Schema: ``segment_weights: {category: {segment: [w_s,w_c,w_rf], _default: [...]}}``."""
    seg_w: Dict[str, dict] = {}
    for (cat, seg), v in fitted.items():
        seg_w.setdefault(str(cat), {})[str(seg)] = [round(float(x), 4) for x in v["weights"]]
    for cat in seg_w:                                  # unseen segments at LIVE -> pure LEGO
        seg_w[cat].setdefault("_default", [0.0, 0.0, 1.0])
    out = out_path or str(Path(config_root) / f"{market}_ensemble_weights.yaml")
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    yaml.safe_dump({"mode": "segment", "segment_weights": seg_w,
                    "default_weights": [0.0, 0.0, 1.0]}, open(out, "w"), sort_keys=True)
    nstack = sum(1 for v in fitted.values() if v["use"] == "STACK")
    logger.info("[weight_fit] wrote %s (segment mode): %d (cat×seg) groups, %d STACK / %d LEGO",
                out, len(fitted), nstack, len(fitted) - nstack)
    return out


def _write_weights_yaml(fitted: Dict[str, dict], market: str, config_root: str,
                        out_path: Optional[str]) -> str:
    """Serialize fitted per-category weights to ``{market}_ensemble_weights.yaml`` in
    the schema ``uk_engine._load_weights`` reads."""
    weights = {cat: {"global": v["weights"][0], "local": v["weights"][1], "lego": v["weights"][2]}
               for cat, v in fitted.items()}
    out = out_path or str(Path(config_root) / f"{market}_ensemble_weights.yaml")
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    yaml.safe_dump({"weights": weights,
                    "default_weights": {"global": 0.0, "local": 0.0, "lego": 1.0},
                    "short_circuit_lego_only": True}, open(out, "w"), sort_keys=True)
    logger.info("[weight_fit] wrote %s", out)
    for cat, v in fitted.items():
        logger.info("  %-26s %-5s w=%s  acc %0.3f->%0.3f  bias %+0.3f->%+0.3f",
                    cat, v["use"], v["weights"], v["acc_lego"], v["acc_stack"],
                    v["bias_lego"], v["bias_stack"])
    return out


def fit_weights(
    market: str,
    *,
    config_root: str = "config",
    project_root: str = ".",
    out_path: Optional[str] = None,
    train_snapshots: Optional[Sequence[str]] = None,
    parent_dir_template: Optional[str] = None,   # DT⋈FS panel (production fidelity)
    panel_path_template: Optional[str] = None,   # OR a pre-joined panel (local datacache)
    lego_path_template: Optional[str] = None,
    actuals_source: Optional[str] = None,
    in_scope_categories: Optional[Sequence[str]] = None,
    components_out_dir: Optional[str] = None,
) -> str:
    """Convenience one-shot FIT: Stage A (save components for every training snapshot)
    then Stage B (fit from the saved components). Used by ``run_stack.py --mode fit``
    and local validation. The Databricks notebook calls the two stages explicitly so
    the saved DIQ output is visible/inspectable between them."""
    cfg, mio, wf = _wf_params(market, config_root)
    snaps = list(train_snapshots or wf["train_snapshots"])
    out_dir = (components_out_dir or wf.get("components_out_dir")
               or str(Path(tempfile.gettempdir()) / f"diq_fit_{market}"))
    for snap in snaps:
        save_components(market, snap, out_dir=out_dir, config_root=config_root,
                        parent_dir_template=parent_dir_template,
                        panel_path_template=panel_path_template,
                        lego_path_template=lego_path_template,
                        in_scope_categories=in_scope_categories)
    return fit_from_components(market, components_dir=out_dir, actuals_source=actuals_source,
                               config_root=config_root, out_path=out_path)
