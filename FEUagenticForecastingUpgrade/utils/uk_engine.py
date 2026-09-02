"""
UK forecasting engine for Databricks production — the integration point that
replaces ``run_full_deterministic_pipeline`` for routed UK categories inside
``utils.diq_runner.run_diq_forecast``.

Architecture (single batch, lazy-initialised shared state)
----------------------------------------------------------
``utils.diq_runner.run_diq_forecast`` calls into this module once per category
inside its per-category loop. The FIRST call within a single ``run_diq_forecast``
invocation does the heavy lifting:

  1. Build feature matrix (lags / rollings / SPLY / promo / holiday) ONCE on the
     full pooled panel (``full_df``, all routed categories together).
  2. Stage the feature matrix to ``/tmp`` parquet so worker processes
     memory-map it instead of pickling 5-10 GB each.
  3. Launch a ``ProcessPoolExecutor`` of ``UK_ENGINE_WORKERS`` workers.
     Each worker trains ONE LightGBM model (single-thread, deterministic) for
     one ``(kind, horizon, category)`` task. Tasks =
        - 13 global fits (one per horizon)
        - N_routed_cats * 13 local fits (one per (cat, horizon))
        - Cats whose frozen weights are (0, 0, 1) -> LEGO only are skipped.
  4. Read ``prediction_rf`` from ``{lego_source}``.
  5. Blend ``DIQ = w_g*global + w_l*local + w_rf*lego`` per category using
     frozen weights from ``config/uk_ensemble_weights.yaml``.
  6. Cache the resulting ``{category -> DataFrame}`` map.

Each subsequent per-category call from ``diq_runner`` just looks up that
category's slice and writes ``inference_forecast.csv`` in the schema
``utils.diq_runner._stack_and_moq`` expects.

Determinism
-----------
LightGBM is deterministic ONLY single-threaded; we run every fit with
a FIXED ``num_threads`` (UK_ENGINE_THREADS, default 4 — LightGBM's
``deterministic=true`` is reproducible across runs when the thread count is
held constant) inside a fresh ``spawn``-started subprocess with seeded
numpy. The result is bit-identical across runs and across clusters as long
as the LightGBM version is pinned. Multi-process parallelism gives us cluster
throughput without sacrificing reproducibility.

Public entry: ``run_uk_engine(cfg, full_df, *, category, snapshot_week, ...)``
"""
from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen artefacts + constants
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS_PATH = "config/uk_ensemble_weights.yaml"
DEFAULT_CONFIG_PATH = "config/config_uk_base_local.yaml"
DEFAULT_LEGO_KEY_COL = "key"
DEFAULT_LEGO_WEEK_COL = "year_week"
DEFAULT_LEGO_PRED_COL = "prediction_rf"
DEFAULT_LEGO_CAT_COL = "category_name"

# inference_forecast.csv schema expected by utils.diq_runner._stack_and_moq.
# We populate every column the historic GBM pipeline emits so downstream
# consumers (MOQ, _stack_and_moq, the client Excel exporter) keep working.
HISTORIC_CSV_COLUMNS = [
    "key", "year_week", "predicted", "actual",
    "origin_period", "forecast_step", "lag",
    "model_level", "model_name", "model_params",
    "is_new_key", "is_dead_key",
]

# Module-level cache: { (snapshot_week, frozenset(routed_cats)) -> state dict }
# State dict = {"per_cat_forecasts": {cat_name: DataFrame}, "n_keys": int, ...}
_RUN_CACHE: Dict[tuple, dict] = {}


# ---------------------------------------------------------------------------
# Category-name normalisation (matches utils/diq_runner.py:1164)
# ---------------------------------------------------------------------------
def _normalise_token(s: str) -> str:
    """Same normalisation utils.diq_runner._filter_and_save_category uses.
    Maps 'DEODORANT & FRAGRANCE' <-> 'DEODORANT_FRAGRANCE'."""
    import re
    s = str(s).upper().replace("&", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _resolve_category_to_human(category_token: str, available_cats: Sequence[str]) -> Optional[str]:
    """Map a widget token (e.g. 'DEODORANT_FRAGRANCE') back to the human form
    ('DEODORANT & FRAGRANCE') found in the panel's category_name column."""
    target = _normalise_token(category_token)
    for c in available_cats:
        if _normalise_token(c) == target:
            return c
    return None


# ---------------------------------------------------------------------------
# Frozen-weights + LEGO loaders
# ---------------------------------------------------------------------------
def _resolve_repo_path(p: str) -> str:
    """Resolve a possibly-relative path against the repo root.

    In a Databricks notebook job the process cwd is /databricks/driver, NOT the
    repo, so a relative default like 'config/uk_ensemble_weights.yaml' would
    FileNotFoundError. Anchor it at the repo root (parent of utils/) instead.
    Absolute paths and cwd-resolvable paths are honoured as-is."""
    pp = Path(p)
    if pp.is_absolute() or pp.exists():
        return str(pp)
    cand = Path(__file__).resolve().parents[1] / p
    return str(cand) if cand.exists() else str(pp)


def _load_weights(weights_path: str) -> dict:
    """Load {weights: {cat: (g,l,rf)}, default_weights: (g,l,rf), short_circuit_lego_only: bool}."""
    with open(_resolve_repo_path(weights_path)) as fh:
        data = yaml.safe_load(fh)
    weights = {k: (v["global"], v["local"], v["lego"]) for k, v in (data.get("weights") or {}).items()}
    default = data.get("default_weights", {"global": 0.0, "local": 0.0, "lego": 1.0})
    return {
        "weights": weights,
        "default": (default["global"], default["local"], default["lego"]),
        "short_circuit_lego_only": bool(data.get("short_circuit_lego_only", True)),
    }


def _read_lego_delta(
    lego_source: str,
    *,
    key_col: str,
    week_col: str,
    pred_col: str,
    cat_col: str,
) -> pd.DataFrame:
    """Read a Delta table, honouring the transaction log so tombstoned/rewritten
    parquet files are excluded.  Tries Spark first (available on Databricks),
    falls back to the ``deltalake`` Python package, then to manual _delta_log
    parsing as a last resort."""
    wanted = [key_col, week_col, pred_col, cat_col]

    # --- Spark path (Databricks) — preferred, always correct ---
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.read.format("delta").load(str(lego_source))
        present = [c for c in wanted if c in sdf.columns]
        if present:
            logger.info("[uk_engine] Delta read via Spark: %s (%d cols)", lego_source, len(present))
            return sdf.select(*present).toPandas()
    except Exception as e:
        logger.warning("[uk_engine] Spark Delta read failed (%s), trying deltalake package", e)

    # --- deltalake package fallback (local dev / off-cluster) ---
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(str(lego_source))
        cols = dt.schema().to_pyarrow().names
        present = [c for c in wanted if c in cols]
        logger.info("[uk_engine] Delta read via deltalake pkg: %s (%d cols)", lego_source, len(present))
        return dt.to_pandas(columns=present if present else None)
    except Exception as e:
        logger.warning("[uk_engine] deltalake package failed (%s), falling back to manual log parse", e)

    # --- Last resort: parse _delta_log manually to find active parquet files ---
    import json
    import pyarrow.parquet as pq
    log_dir = os.path.join(str(lego_source), "_delta_log")
    adds, removes = set(), set()
    for f in sorted(os.listdir(log_dir)):
        if not f.endswith(".json"):
            continue
        for line in open(os.path.join(log_dir, f)):
            entry = json.loads(line)
            if "add" in entry:
                adds.add(entry["add"]["path"])
            if "remove" in entry:
                removes.add(entry["remove"]["path"])
    active = [os.path.join(str(lego_source), p) for p in sorted(adds - removes)]
    if not active:
        raise RuntimeError(f"No active parquet files in Delta log at {lego_source}")
    cols = pq.ParquetFile(active[0]).schema_arrow.names
    present = [c for c in wanted if c in cols]
    logger.info("[uk_engine] Delta read via manual log parse: %s (%d active files, %d cols)",
                lego_source, len(active), len(present))
    return pd.concat([pq.read_table(f, columns=present or None).to_pandas()
                       for f in active], ignore_index=True)


def _read_lego(
    lego_source: str,
    *,
    key_col: str = DEFAULT_LEGO_KEY_COL,
    week_col: str = DEFAULT_LEGO_WEEK_COL,
    pred_col: str = DEFAULT_LEGO_PRED_COL,
    cat_col: str = DEFAULT_LEGO_CAT_COL,
) -> pd.DataFrame:
    """Read prediction_rf from a parquet or Delta directory. Returns
    DataFrame[key, year_week, prediction_rf, category_name] with string keys.

    Delta-aware: if the source directory contains a ``_delta_log/`` subfolder
    the read honours the transaction log (via Spark, deltalake pkg, or manual
    log parsing) so tombstoned/rewritten parquet files are excluded."""
    logger.info("[uk_engine] reading LEGO prediction_rf from %s", lego_source)

    # ---- Delta-aware read: honour the transaction log ----
    is_delta = os.path.isdir(os.path.join(str(lego_source), "_delta_log"))
    if is_delta:
        logger.info("[uk_engine] detected Delta format at %s", lego_source)
        df = _read_lego_delta(lego_source, key_col=key_col, week_col=week_col,
                              pred_col=pred_col, cat_col=cat_col)
    else:
        # plain parquet directory — existing pyarrow path
        import pyarrow.dataset as pads
        schema = pads.dataset(lego_source).schema
        cols_present = schema.names
        wanted = [c for c in (key_col, week_col, pred_col, cat_col) if c in cols_present]
        if not wanted:
            raise RuntimeError(
                f"uk_engine: LEGO source {lego_source!r} has none of the expected columns "
                f"({key_col}, {week_col}, {pred_col}, {cat_col}). Found: {cols_present[:12]}"
            )
        df = pd.read_parquet(lego_source, columns=wanted)

    if key_col in df.columns:
        df[key_col] = df[key_col].astype(str)
    if week_col in df.columns:
        df[week_col] = df[week_col].astype(str).str.replace("-", "", regex=False)
    if pred_col in df.columns:
        df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0)
    df = df.drop_duplicates([key_col, week_col])
    logger.info("[uk_engine] LEGO: %d rows, %d unique keys, %d unique weeks",
                len(df), df[key_col].nunique(), df[week_col].nunique())
    return df


# ---------------------------------------------------------------------------
# Worker process — one LightGBM fit + predict
# ---------------------------------------------------------------------------
def _worker_init(project_root: str) -> None:
    """ProcessPoolExecutor initializer. Spawned workers get a fresh sys.path that
    doesn't inherit the parent's, so we must push the repo root in before any
    `from utils.uk_forecast import ...` happens inside the worker function."""
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# Per-worker-process cache of the staged panel. A worker handles many tasks
# (156 tasks / ~7 workers ≈ 22 each); without this every task re-reads the
# multi-GB parquet from /tmp, and N concurrent readers thrash the disk.
_WORKER_PANEL_CACHE: dict = {}

# Diagnostic-only: when env UK_ENGINE_DEBUG_MERGED is set, _prepare_state stashes
# the per-category g/l/rf/predicted blend frame here for offline inspection.
# Inert in production (gated on the env var). Safe to leave; written only in main proc.
_DEBUG_MERGED: dict = {}


def _train_and_predict_one(args: tuple) -> dict:
    """Worker entry. Trains one LightGBM model (UK_ENGINE_THREADS threads,
    deterministic) and returns a small dict with the forecast-row predictions.

    args = (kind, horizon, category, panel_path, train_end, snapshot_week,
            target_col, key_col, week_col, cat_col, feat_cols, cat_cols, lgb_params)
    """
    (kind, h, category, panel_path, train_end, snapshot_week,
     target_col, key_col, week_col, cat_col, feat_cols, cat_cols, lgb_params) = args
    import lightgbm as lgb

    # Read staged panel ONCE per worker process and reuse across tasks.
    cols = list({key_col, week_col, target_col, cat_col, "_wi", *feat_cols})
    cache_key = (panel_path, tuple(sorted(c for c in cols if c is not None)))
    df = _WORKER_PANEL_CACHE.get(cache_key)
    if df is None:
        df = pd.read_parquet(panel_path, columns=[c for c in cols if c is not None])
        _WORKER_PANEL_CACHE.clear()          # one panel per worker is plenty
        _WORKER_PANEL_CACHE[cache_key] = df

    # Filter to category if local. NOTE: produces a new frame — the cached panel
    # is never mutated (all downstream ops build new left/right frames).
    if kind == "local" and cat_col in df.columns:
        df = df[df[cat_col] == category]

    if df.empty:
        return {"kind": kind, "horizon": h, "category": category,
                "predictions": pd.DataFrame(columns=[key_col, week_col, "value"])}

    # Build (origin_week -> target_week) pairs: target_week = origin_week + h
    # Vectorised via a unique-week add-h map (only ~150 distinct weeks in the panel).
    from utils.uk_forecast import add_weeks
    weeks = sorted(df[week_col].unique())
    tw_map = {w: add_weeks(w, h) for w in weeks}
    left = df[[key_col, week_col, "_wi", *feat_cols]].copy()
    left["_tw"] = left[week_col].map(tw_map)
    right = df[[key_col, week_col, target_col]].rename(columns={week_col: "_tw", target_col: "_y"})
    pairs = left.merge(right, on=[key_col, "_tw"], how="inner")

    tr = pairs[pairs["_tw"] <= train_end]
    pr = pairs[pairs[week_col] == train_end]   # forecast origin -> _tw = train_end + h

    if tr.empty or pr.empty:
        return {"kind": kind, "horizon": h, "category": category,
                "predictions": pd.DataFrame(columns=[key_col, week_col, "value"])}

    # Recency weight (utils/uk_forecast.py:222 pattern)
    te_rank = tr["_wi"].max()
    halflife = float(lgb_params.pop("_recency_halflife_weeks", 39))
    w = np.power(0.5, (te_rank - tr["_wi"]).clip(lower=0) / halflife)

    nb = int(lgb_params.pop("_num_boost_round", 700))
    dtrain = lgb.Dataset(tr[feat_cols], label=np.clip(tr["_y"].values, 0, None),
                         weight=w.values, categorical_feature=cat_cols, free_raw_data=False)
    model = lgb.train(lgb_params, dtrain, num_boost_round=nb)
    yhat = model.predict(pr[feat_cols])
    out = pr[[key_col, "_tw"]].rename(columns={"_tw": week_col}).copy()
    out["value"] = np.clip(yhat, 0, None)
    return {"kind": kind, "horizon": h, "category": category, "predictions": out}


# ---------------------------------------------------------------------------
# State preparation — runs once per snapshot
# ---------------------------------------------------------------------------
def _prepare_state(
    cfg, full_df: pd.DataFrame, *, snapshot_week: str,
    route_categories: Sequence[str], weights_path: str, lego_source: str,
    config_path: str = DEFAULT_CONFIG_PATH, lego_pred_col: str = DEFAULT_LEGO_PRED_COL,
    return_components: bool = False,
) -> dict:
    """Heavy one-time work for a snapshot. Returns
    {"per_cat_forecasts": {category_human_name: DataFrame[key, year_week, predicted]}}.

    ``config_path`` / ``lego_pred_col`` select the market: UK defaults to
    config_uk_base_local.yaml + prediction_rf; DE passes config_de_base_local.yaml
    + prediction_xgb."""
    from utils.uk_forecast import (
        load_config, add_weeks, add_backward_features,
        _encode_categoricals, _future_cols, _lgb_params,
    )

    # ----- 0. Load model config + weights -----
    model_cfg = load_config(_resolve_repo_path(config_path), overrides={
        "input_source": "dbx",            # marker; we don't call read_input_panel
        "collapse_promo": False,           # off in v1 production (A/B was +0.001)
        "holiday_distance_features": False, # off in v1 production (A/B was -0.01)
        "rich_features": False,            # off in v1 production
        "per_category": False,             # we drive global/local manually
        "bias_calibration": False,         # baked into weights pool
    })
    weights = _load_weights(weights_path)

    # ----- 1. Resolve routed categories from tokens to human names in the panel -----
    if "category_name" in full_df.columns:
        avail_cats = full_df["category_name"].astype(str).unique().tolist()
    elif "Category" in full_df.columns:
        avail_cats = full_df["Category"].astype(str).unique().tolist()
    else:
        avail_cats = []
    routed_human = [r for r in (
        _resolve_category_to_human(c, avail_cats) for c in route_categories) if r]
    if not routed_human:
        raise RuntimeError(
            f"uk_engine: no routed category resolved against panel categories. "
            f"requested={route_categories}, panel={avail_cats[:8]}"
        )
    logger.info("[uk_engine] routed categories resolved: %s", routed_human)

    # Which cats need only LEGO (skip global+local training entirely)?
    lego_only = set()
    if weights["short_circuit_lego_only"]:
        for c in routed_human:
            (wg, wl, wrf) = weights["weights"].get(c, weights["default"])
            if abs(wg) < 1e-9 and abs(wl) < 1e-9 and abs(wrf - 1.0) < 1e-9:
                lego_only.add(c)
    train_cats = [c for c in routed_human if c not in lego_only]
    logger.info("[uk_engine] training local models for %d cats; LEGO-only short-circuit for %d cats: %s",
                len(train_cats), len(lego_only), sorted(lego_only))

    # ----- 2. Subset the panel to routed cats + sort -----
    cat_col = "category_name" if "category_name" in full_df.columns else "Category"
    df = full_df[full_df[cat_col].astype(str).isin(routed_human)].copy()
    df = df.sort_values(["key", "year_week"], kind="mergesort").reset_index(drop=True)
    df["year_week"] = df["year_week"].astype(str).str.replace("-", "", regex=False)
    df[model_cfg["target"]] = pd.to_numeric(df[model_cfg["target"]], errors="coerce").fillna(0.0)

    # CRITICAL: _encode_categoricals (below) converts category_name to int codes
    # IN PLACE because it's a model feature. Every downstream STRING comparison
    # (local-task filter, per-cat key sets, LEGO category slice) must therefore
    # use this untouched raw copy — comparing codes to 'MINI MEAL' silently
    # matches nothing and yields 0-row forecasts everywhere.
    RAW_CAT = "_cat_raw"
    df[RAW_CAT] = df[cat_col].astype(str)

    # ----- 3. Build features (vectorised) -----
    # Override the cat_col in cfg so feature helpers find it
    model_cfg = {**model_cfg, "cat_col": cat_col}
    back_cols = add_backward_features(df, model_cfg)
    cat_cols = _encode_categoricals(df, model_cfg)
    fut_cols = _future_cols(df, model_cfg)
    feat_cols = back_cols + cat_cols + fut_cols
    weeks = sorted(df["year_week"].unique())
    rank = {w: i for i, w in enumerate(weeks)}
    df["_wi"] = df["year_week"].map(rank).astype("int32")
    train_end = add_weeks(snapshot_week, -1)

    # ----- 4. Stage panel to /tmp parquet -----
    panel_path = Path(tempfile.gettempdir()) / f"uk_engine_panel_{snapshot_week}.parquet"
    keep = list(dict.fromkeys([model_cfg["key_col"], "year_week", "_wi",
                               model_cfg["target"], RAW_CAT, *feat_cols]))
    df[keep].to_parquet(panel_path, index=False)
    logger.info("[uk_engine] staged feature panel: %s (%.1f MB)",
                panel_path, panel_path.stat().st_size / 1e6)

    # ----- 5. Build LightGBM params (single-thread, deterministic) -----
    lgb_params = _lgb_params(model_cfg)
    # num_threads is set per-task below (UK_ENGINE_THREADS, default 4, fixed across
    # runs => reproducible per LightGBM's deterministic=true contract).
    lgb_params["_num_boost_round"] = model_cfg.get("n_estimators", 700)
    lgb_params["_recency_halflife_weeks"] = float(model_cfg.get("recency_halflife_weeks", 39))

    horizons = list(range(1, int(getattr(cfg, "forecast_horizon", 13)) + 1))
    # Workers filter local tasks by RAW_CAT (string), NOT the encoded cat_col.
    common = (str(panel_path), train_end, str(snapshot_week),
              model_cfg["target"], model_cfg["key_col"], "year_week", RAW_CAT,
              feat_cols, cat_cols)

    g_tasks: List[tuple] = [("global", h, None, *common, dict(lgb_params)) for h in horizons]
    l_tasks: List[tuple] = [("local", h, c, *common, dict(lgb_params))
                            for c in train_cats for h in horizons]
    # INTERLEAVE globals among locals. Global tasks train on the FULL pooled panel
    # (multi-GB merge each); if all 13 hit the pool first, N concurrent global
    # merges blow past driver RAM and the run swap-thrashes. Spacing them out
    # keeps at most ~1-2 globals in flight at any moment.
    tasks: List[tuple] = []
    if g_tasks and l_tasks:
        step = max(1, len(l_tasks) // len(g_tasks))
        gi = 0
        for i, lt in enumerate(l_tasks):
            if i % step == 0 and gi < len(g_tasks):
                tasks.append(g_tasks[gi]); gi += 1
            tasks.append(lt)
        tasks.extend(g_tasks[gi:])
    else:
        tasks = g_tasks + l_tasks
    logger.info("[uk_engine] %d tasks to train (%d global + %d local, interleaved)",
                len(tasks), len(g_tasks), len(l_tasks))

    # ----- 6. Train + predict in parallel -----
    # Threads-per-fit x workers ≈ driver cores. LightGBM deterministic=True is
    # reproducible across runs as long as num_threads is FIXED (LightGBM docs:
    # same data + same params incl. num_threads). 4 threads/fit makes the big
    # global fits ~4x faster than single-thread; fewer concurrent workers keeps
    # peak memory bounded (each worker caches the panel + builds one merge).
    threads = int(os.environ.get("UK_ENGINE_THREADS", "4"))
    for t in tasks:
        t[-1]["num_threads"] = threads   # mutate each task's own params dict
    default_workers = max(1, (os.cpu_count() or 8) // threads - 1)
    n_workers = int(os.environ.get("UK_ENGINE_WORKERS", default_workers))
    logger.info("[uk_engine] launching ProcessPoolExecutor: max_workers=%d, "
                "num_threads/fit=%d (override via UK_ENGINE_WORKERS / UK_ENGINE_THREADS)",
                n_workers, threads)
    import multiprocessing as mp
    import time as _time
    ctx = mp.get_context("spawn")  # clean interpreter per worker; deterministic startup
    # Spawn-mode workers get a fresh sys.path that does NOT inherit the parent's.  On Databricks
    # this means `from utils.uk_forecast import add_weeks` will fail inside the worker unless
    # we push the project root onto sys.path at the very top of each worker.
    project_root = str(Path(__file__).resolve().parents[1])
    global_preds: Dict[int, pd.DataFrame] = {}
    local_preds: Dict[tuple, pd.DataFrame] = {}
    t0 = _time.time()
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx,
        initializer=_worker_init, initargs=(project_root,),
    ) as pool:
        futs = {pool.submit(_train_and_predict_one, t): (t[0], t[1], t[2]) for t in tasks}
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            if res["kind"] == "global":
                global_preds[res["horizon"]] = res["predictions"]
            else:
                local_preds[(res["category"], res["horizon"])] = res["predictions"]
            done += 1
            if done % 10 == 0 or done == len(tasks):
                logger.info("[uk_engine] trained %d/%d models (%.1f min elapsed)",
                            done, len(tasks), (_time.time() - t0) / 60)
    logger.info("[uk_engine] training complete: %d global / %d local prediction frames (%.1f min)",
                len(global_preds), len(local_preds), (_time.time() - t0) / 60)

    # ----- 7. Read LEGO + frozen weights -----
    lego_df = _read_lego(lego_source, pred_col=lego_pred_col)
    # Attach category to LEGO (for blending per cat). If the LEGO panel has its own category col,
    # we trust it; else we derive from the main panel.
    if DEFAULT_LEGO_CAT_COL not in lego_df.columns:
        catmap = df.drop_duplicates(model_cfg["key_col"]).set_index(model_cfg["key_col"])[RAW_CAT]
        lego_df[DEFAULT_LEGO_CAT_COL] = lego_df[DEFAULT_LEGO_KEY_COL].map(catmap)

    # ----- 8. Blend per category -----
    per_cat_forecasts: Dict[str, pd.DataFrame] = {}
    components: Dict[str, pd.DataFrame] = {}   # per-cat [key,year_week,horizon,g,l,rf,predicted] for FIT mode
    for c in routed_human:
        (wg, wl, wrf) = weights["weights"].get(c, weights["default"])
        # Gather global + local predictions for this cat across all horizons
        gframes, lframes = [], []
        for h in horizons:
            gp = global_preds.get(h)
            if gp is not None and not gp.empty:
                gp_c = gp.copy()
                # Filter global predictions to keys in this category
                cat_keys = set(df[df[RAW_CAT] == c][model_cfg["key_col"]].astype(str).unique())
                gp_c = gp_c[gp_c[model_cfg["key_col"]].astype(str).isin(cat_keys)]
                gp_c["horizon"] = h
                gframes.append(gp_c.rename(columns={"value": "g"}))
            lp = local_preds.get((c, h))
            if lp is not None and not lp.empty:
                lp_c = lp.copy()
                lp_c["horizon"] = h
                lframes.append(lp_c.rename(columns={"value": "l"}))
        g_df = pd.concat(gframes, ignore_index=True) if gframes else pd.DataFrame(
            columns=[model_cfg["key_col"], "year_week", "g", "horizon"])
        l_df = pd.concat(lframes, ignore_index=True) if lframes else pd.DataFrame(
            columns=[model_cfg["key_col"], "year_week", "l", "horizon"])
        merged = g_df.merge(l_df, on=[model_cfg["key_col"], "year_week", "horizon"], how="outer")
        merged["g"] = merged["g"].fillna(0.0)
        merged["l"] = merged["l"].fillna(0.0)
        # Join LEGO
        lego_c = lego_df[lego_df[DEFAULT_LEGO_CAT_COL] == c][
            [DEFAULT_LEGO_KEY_COL, DEFAULT_LEGO_WEEK_COL, lego_pred_col]].rename(
            columns={DEFAULT_LEGO_KEY_COL: model_cfg["key_col"],
                     DEFAULT_LEGO_WEEK_COL: "year_week",
                     lego_pred_col: "rf"})
        merged = merged.merge(lego_c, on=[model_cfg["key_col"], "year_week"], how="left")
        merged["rf"] = merged["rf"].fillna(0.0)
        merged["predicted"] = np.clip(wg * merged["g"] + wl * merged["l"] + wrf * merged["rf"], 0, None)
        merged["forecast_step"] = merged["horizon"]
        merged["lag"] = merged["horizon"] - 1
        per_cat_forecasts[c] = merged[[model_cfg["key_col"], "year_week",
                                       "predicted", "forecast_step", "lag", "horizon"]]
        if return_components or os.environ.get("UK_ENGINE_DEBUG_MERGED"):
            components[c] = merged[[model_cfg["key_col"], "year_week", "horizon",
                                    "g", "l", "rf", "predicted"]].copy()
            if os.environ.get("UK_ENGINE_DEBUG_MERGED"):
                _DEBUG_MERGED[c] = merged.copy()
        logger.info("[uk_engine] %s: blended %d (key, year_week) rows (g=%.2f, l=%.2f, rf=%.2f)",
                    c, len(merged), wg, wl, wrf)

    return {
        "per_cat_forecasts": per_cat_forecasts,
        "components": components,        # populated only when return_components=True (FIT mode)
        "model_cfg": model_cfg,
        "lego_only": lego_only,
        "panel_path": str(panel_path),
    }


# ---------------------------------------------------------------------------
# Per-category output writer (matches GBM CSV schema for _stack_and_moq)
# ---------------------------------------------------------------------------
def _write_per_category_csv(cfg, snapshot_week: str, forecast: pd.DataFrame, model_cfg: dict) -> Path:
    """Write inference_forecast.csv with the historic column set so
    utils.diq_runner._stack_and_moq (which does pd.read_csv → concat → MOQ)
    consumes it transparently."""
    out_dir = Path(cfg.artifact_base_path) / "model_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_forecast.csv"

    out = pd.DataFrame()
    out["key"] = forecast[model_cfg["key_col"]].astype(str)
    out["year_week"] = forecast["year_week"].astype(str)
    out["predicted"] = forecast["predicted"].astype("float64")
    out["actual"] = np.nan
    out["origin_period"] = str(snapshot_week)
    out["forecast_step"] = forecast["forecast_step"].astype("Int32")
    out["lag"] = forecast["lag"].astype("Int32")
    out["model_level"] = "stacked_ensemble"
    out["model_name"] = "uk_stacked_v1"
    out["model_params"] = "rf-ensemble(global+local+lego), frozen 7-origin weights"
    out["is_new_key"] = False
    out["is_dead_key"] = False
    out = out[HISTORIC_CSV_COLUMNS]
    out.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Public entry — called per category from utils.diq_runner.run_diq_forecast
# ---------------------------------------------------------------------------
def run_uk_engine(
    cfg,
    full_df: pd.DataFrame,
    *,
    category: str,
    snapshot_week: str,
    route_categories: Sequence[str],
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    lego_source: Optional[str] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
    lego_pred_col: str = DEFAULT_LEGO_PRED_COL,
    num_threads: int = 1,   # for the historical contract; ignored — workers use 1
) -> Path:
    """Per-category entry. On first call within a single ``run_diq_forecast``
    invocation, this builds features + trains all models + reads LEGO once.
    Subsequent calls for sibling categories reuse the cached state and only
    write that category's ``inference_forecast.csv``.

    Parameters mirror the existing overlay convention (``full_df``,
    ``snapshot_week``, ``route_categories``) so adding this to ``diq_runner``
    is surgical (an ``if`` branch in the per-cat loop).
    """
    if not route_categories:
        raise ValueError("uk_engine: route_categories must be non-empty")
    if not lego_source:
        raise ValueError(
            "uk_engine: lego_source is required (path to TF_modelling_pristine / "
            "or any Spark-parquet dir carrying key, year_week, prediction_rf, "
            "category_name)"
        )

    # Normalise ONCE at the boundary: the Databricks widget passes '2026-18',
    # all internal week arithmetic + panel joins use compact '202618'.
    snapshot_week = str(snapshot_week).replace("-", "")

    key = (snapshot_week, config_path, lego_pred_col,
           frozenset(_normalise_token(c) for c in route_categories))
    if key not in _RUN_CACHE:
        logger.info("=" * 70)
        logger.info("[uk_engine] FIRST call this snapshot — initialising shared state")
        logger.info("[uk_engine]   snapshot=%s  routed=%d cats", snapshot_week, len(route_categories))
        logger.info("=" * 70)
        _RUN_CACHE[key] = _prepare_state(
            cfg=cfg, full_df=full_df, snapshot_week=str(snapshot_week),
            route_categories=route_categories, weights_path=weights_path,
            lego_source=lego_source, config_path=config_path, lego_pred_col=lego_pred_col,
        )

    state = _RUN_CACHE[key]
    avail_cats = list(state["per_cat_forecasts"].keys())
    human = _resolve_category_to_human(category, avail_cats)
    if human is None:
        raise RuntimeError(
            f"uk_engine: category {category!r} resolved against routed set "
            f"{sorted(route_categories)} produced no match. Available human "
            f"names: {avail_cats}"
        )

    forecast = state["per_cat_forecasts"][human]
    if forecast.empty:
        # FAIL LOUDLY. A 0-row forecast means the panel/blend produced nothing
        # for this category — writing an empty CSV would let the run report
        # success while shipping no forecast (exactly the silent failure mode
        # we hit on 2026-06-10). diq_runner's continue_on_category_failure
        # turns this into a visible per-category FAILED instead.
        raise RuntimeError(
            f"uk_engine: blended forecast for {human!r} is EMPTY — check that the "
            f"input panel has rows for this category at the forecast origin "
            f"(snapshot-1) and across the {getattr(cfg, 'forecast_horizon', 13)}-week "
            f"window. Refusing to write a 0-row inference_forecast.csv."
        )

    out_path = _write_per_category_csv(
        cfg=cfg, snapshot_week=snapshot_week,
        forecast=forecast,
        model_cfg=state["model_cfg"],
    )
    logger.info("[uk_engine] %s -> %s (%d rows)",
                human, out_path, len(state["per_cat_forecasts"][human]))
    return out_path


def compute_components(
    cfg,
    full_df: pd.DataFrame,
    *,
    snapshot_week: str,
    route_categories: Sequence[str],
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    lego_source: Optional[str] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
    lego_pred_col: str = DEFAULT_LEGO_PRED_COL,
) -> pd.DataFrame:
    """FIT-mode entry: train the engine for one snapshot and return the raw
    blend components — one row per (key, year_week, horizon) with columns
    ``[key, year_week, horizon, g, l, rf, predicted, category]`` — instead of
    writing per-category CSVs. Used by ``utils.weight_fit`` to estimate the
    per-category ensemble weights across rolling-origin snapshots.

    Deliberately bypasses the ``_RUN_CACHE``/CSV-writing LIVE path so it has zero
    effect on ``run_uk_engine`` (the production forecast path)."""
    if not lego_source:
        raise ValueError("compute_components: lego_source is required")
    snapshot_week = str(snapshot_week).replace("-", "")
    state = _prepare_state(
        cfg=cfg, full_df=full_df, snapshot_week=snapshot_week,
        route_categories=route_categories, weights_path=weights_path,
        lego_source=lego_source, config_path=config_path, lego_pred_col=lego_pred_col,
        return_components=True,
    )
    frames = []
    for cat, comp in state["components"].items():
        cc = comp.copy(); cc["category"] = cat
        frames.append(cc)
    if not frames:
        raise RuntimeError(f"compute_components: no components produced for snapshot {snapshot_week}")
    return pd.concat(frames, ignore_index=True)


__all__ = ["run_uk_engine", "compute_components"]
