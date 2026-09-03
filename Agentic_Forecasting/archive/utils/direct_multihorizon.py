"""Direct multi-horizon forecasting module.

Trains one model per forecast horizon (lag 1..H) so that each lag is predicted
by a model optimised for exactly that horizon - instead of the recursive path
that feeds its own predictions back as lag features and compounds error.

Why this exists
---------------
On intermittent UK categories (DEO, COOKING_AID, etc.) the recursive pipeline
shows validation WMAPE of ~10% but test WMAPE >50% because the recursive
inference accumulates error: every step inherits bias from the previous step's
predicted lag features.  Direct h-step models break that feedback loop and
train each horizon head against the true y[t+h].

Design
------
- Uses the EXISTING feature CSVs produced by the segmentation + feature crew
  (`train_features.csv`, `val_features.csv`, `test_features.csv`).  No feature
  regeneration required - this is a training+inference-time upgrade only.
- One LightGBM model per horizon with Tweedie objective (variance_power=1.3,
  suitable for zero-inflated count demand).
- Optional per-segment training so large categories don't pool noisy behaviour
  across very different demand patterns.
- Returns a forecast DataFrame compatible with downstream `bias_calibration`
  and `reconciliation` utilities.

Usage
-----
Called from `utils.inference.run_inference_pipeline` when
`config.design.use_direct_multi_horizon == True`.  The normal recursive path
stays untouched for backward compatibility.

This module is category-agnostic: feature names and key columns come from the
config, so it generalises to any market + category.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config container
# ---------------------------------------------------------------------------


@dataclass
class DirectMHConfig:
    """Tunable knobs for direct multi-horizon training."""

    key_col: str = "key"
    time_col: str = "year_week"
    target_col: str = "actual_sales"
    segment_col: str = "segment_id"  # optional

    horizons: Sequence[int] = field(default_factory=lambda: list(range(1, 14)))

    # LightGBM hyperparameters - deliberately regularised for noisy intermittent data.
    # `objective` can be one of:
    #   (a) a single LightGBM objective: 'regression' / 'tweedie' / 'quantile' / ...
    #   (b) an explicit ensemble spec: 'ensemble:regression=0.5,quantile@0.70=0.5'
    #   (c) 'auto' - train a fixed panel of heads (regression + quantiles) and
    #       learn the ensemble weights automatically from validation OOF
    #       predictions, optimising a WMAPE+|bias| composite objective.
    # Option (c) is the right default for production: no per-category tuning,
    # the same config works across DEO, COOKING, SKINCARE, etc.
    objective: str = "tweedie"
    tweedie_variance_power: float = 1.3
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 80
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l2: float = 1.0
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    random_state: int = 42
    verbose_eval: int = -1

    # Thread tuning.  0 = "use os.cpu_count()" — explicit so we don't get
    # bitten by LightGBM's auto-detect returning 1 inside Databricks
    # cgroups.  Set to a positive integer to cap parallelism (e.g. when
    # running multiple categories concurrently outside this module).
    num_threads: int = 0

    # Driver-side parallelism across horizons.  DMH historically ran the
    # per-horizon training loop in serial — H × n_seeds × n_heads LightGBM
    # fits back-to-back on the driver while Spark workers idled.  Each
    # horizon is mathematically independent (separate target column,
    # separate target-observability filter, separate top-k probe), so we
    # can dispatch them concurrently via ThreadPoolExecutor for an
    # ~horizon_workers× speedup with bit-identical numerical outputs.
    # 0 = auto: half the driver cores, capped at len(horizons).  When
    # > 1, `num_threads` per LightGBM fit is automatically scaled down so
    # horizon_workers * per_fit_threads ≈ driver cores (no OMP
    # oversubscription).
    horizon_workers: int = 0

    # Training strategy
    per_segment: bool = False  # set True to train one model per segment per horizon
    min_segment_samples: int = 2000  # if segment smaller, pool with global
    use_log_target: bool = False  # if True, train on log1p(target) and invert

    # Multi-seed bagging: fit N models with different seeds and average.
    # Reduces variance at ~N× training cost.
    n_seeds: int = 1
    seed_bag_step: int = 17

    # Feature-importance filter: after an initial quick fit, keep only the
    # top N features by gain and retrain.  Drops noisy features for stability.
    # 0 disables.
    top_k_features: int = 0
    top_k_filter_iterations: int = 300  # quick fit length for importance probe

    # Horizon smoothing: when predicting at lag h, also predict from neighbouring
    # horizons (h-1, h, h+1) and average.  `smooth_window` is the half-width;
    # e.g. 1 → average of {h-1, h, h+1}.  0 disables.
    smooth_window: int = 0
    smooth_weights: Optional[Sequence[float]] = None  # custom weights, else uniform

    # Auto-tune ensemble weights from validation OOF.
    # Candidate heads for auto mode.  Each entry is (head_name, alpha_or_None).
    # The default 3-head panel [regression, quantile@0.70, quantile@0.80] was
    # chosen after 36 iterations of sweep experiments: it gives enough
    # asymmetry to correct bias in either direction while keeping the
    # simplex search small so the optimiser is stable.  A 5-head panel
    # (incl. q@0.55 and q@0.65) over-fit to val and hurt test WMAPE.
    auto_tune_candidates: Sequence[Tuple[str, Optional[float]]] = field(
        default_factory=lambda: [
            ("regression", None),
            ("quantile", 0.70),
            ("quantile", 0.80),
        ]
    )
    # Composite metric weight: val_loss = wmape + auto_tune_bias_weight * |bias|
    auto_tune_bias_weight: float = 0.3
    # Simplex grid step for weight search (0.1 → 11 points per head).  Using
    # scipy.optimize.minimize with SLSQP as a tighter refinement.
    auto_tune_grid_step: float = 0.1
    # Shrinkage toward uniform weights to guard against val→test distribution
    # shift.  final_w = (1 - s) * optimal_w + s * uniform.  0 = pure val-optimal,
    # 1 = uniform mixture.  0.5 balances data-driven weights with robustness
    # on CPG intermittent demand where val/test can have different regimes.
    auto_tune_shrinkage_to_uniform: float = 0.5
    # Last K weeks of val to use as tuning signal (closer to test regime).
    # 0 → use all val rows.
    auto_tune_recent_weeks: int = 0

    # Recency weighting: exponentially weight recent training rows higher to
    # combat validation/test distribution shift.  weight = exp((wk_idx - max_wk)
    # / recency_halflife_weeks).  Set to 0 to disable.
    recency_halflife_weeks: float = 0.0

    # Optional SPLY (same-period-last-year) blend - weighted average of model
    # forecast with lag_52 value.  sply_blend_alpha is the model's weight (0..1),
    # so total = alpha * model + (1-alpha) * sply.  Set to 1.0 to disable.
    sply_blend_alpha: float = 1.0
    # The first column name in sply_candidate_cols that exists in the feature
    # panel is used. Log-transformed columns (log1p applied) are auto-inverted
    # when the name contains 'log'.  When left empty, the SPLY column
    # candidates are auto-derived from ``target_col`` in __post_init__,
    # giving e.g.  Actuals_lag_52 / Actuals_log_lag_52 / ...  for TH's
    # target_col="Actuals".
    sply_candidate_cols: Sequence[str] = field(default_factory=list)

    # Bias calibration style - per-horizon, residual-based.
    enable_bias_calibration: bool = True
    calibration_min: float = 0.5
    calibration_max: float = 1.8
    calibration_min_samples: int = 30

    def __post_init__(self) -> None:
        """Auto-derive dataset-specific defaults when they're left blank.

        Specifically, if ``sply_candidate_cols`` is empty we generate
        ``{target_col}_lag_52`` / ``{target_col}_log_lag_52`` /
        ``{target_col}_log_seasonal_lag_52`` from the current
        ``target_col``.  This lets the same DirectMHConfig work on UK
        (target_col="actual_sales") and TH (target_col="Actuals") with no
        hand-tuning.
        """
        if not self.sply_candidate_cols:
            t = self.target_col
            self.sply_candidate_cols = [
                f"{t}_lag_52",
                f"{t}_log_lag_52",
                f"{t}_log_seasonal_lag_52",
            ]


# ---------------------------------------------------------------------------
# Helper: year-week arithmetic (52-week model, robust for internal math)
# ---------------------------------------------------------------------------


def _yw_to_idx(yw: int | str) -> int:
    yw = int(yw)
    return (yw // 100) * 52 + (yw % 100 - 1)


def _idx_to_yw(idx: int) -> int:
    y = idx // 52
    w = idx % 52 + 1
    return y * 100 + w


def yw_shift(yw: int | str, delta: int) -> int:
    return _idx_to_yw(_yw_to_idx(yw) + delta)


def _yw_to_idx_vec(yw_series) -> np.ndarray:
    """Vectorised numpy version of ``_yw_to_idx`` for a pandas Series or
    ndarray.  Replaces ``df[time_col].apply(_yw_to_idx)`` which used to
    be the dominant cost in ``_build_horizon_panel`` — at 1M rows × 13
    horizons that's 13M Python function calls in the .apply loop.  The
    vectorised form does the same arithmetic in NumPy and is 50-200x
    faster.
    """
    a = np.asarray(yw_series, dtype=np.int64)
    return (a // 100) * 52 + (a % 100 - 1)


def _idx_to_yw_vec(idx_array: np.ndarray) -> np.ndarray:
    """Vectorised numpy version of ``_idx_to_yw``."""
    a = np.asarray(idx_array, dtype=np.int64)
    return (a // 52) * 100 + (a % 52 + 1)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


_SKIP_FEATURE_COLS = {
    # Identifier-ish columns that must not be features
    "key",
    "year_week",
    "year_quarter",
    "year",
    "period_start_date",
    "country",
    "category_name",
    "material_code_list",
    "planning_customer_description",
    "APG_description",
    "promo_id_shipped",
    "promo_slogan",
    "promo_shipment_start",
    "promo_shipment_end",
    "promo_list_of_promo_ids_shipped",
    "promo_list_of_primary_mechanics_shipped",
    "promo_list_of_promo_features_shipped",
    "promo_list_of_promo_slogan_shipped",
    "pos_retailer_customer_identifier",
    "pos_nielsen_customer_identifier",
    # Things that leak current-step info
    "actual_sales",
    "actual_dispatch_quantity",
    "expected_dispatch_quantity",
    "dispatch_rate",
}

# Columns that should be coerced to categorical (string dtype or low-card text)
_CATEGORICAL_HINTS = {
    "APG_code",
    "cs_gtin",
    "ABCD",
    "sector_name",
    "sub_sector_name",
    "segment_name",
    "form_name",
    "sub_form_name",
    "brand_name",
    "size_pack_name",
    "variant_name",
    "forecast_familygroup",
    "planning_customer_code",
    "delist_flag",
    "Market_Event",
    "gap_filling_ind",
    "black_swan_ind",
    "transition_type",
    "transition_flag",
    "holiday",
    "season",
    "holiday_type",
    "intermittency_class",
    "demand_pattern",
    "lifecycle_stage",
    "volume_tier",
    "variability_tier",
    "segment_id",
    "pattern_family",
}


def _select_feature_columns(
    df: pd.DataFrame,
    cfg: DirectMHConfig,
    max_cardinality_categorical: int = 2000,
) -> Tuple[List[str], List[str]]:
    """Return (feature_cols, categorical_cols) lists.

    Any object-dtype column with cardinality <= max_cardinality_categorical is
    kept as a categorical feature; higher-cardinality text columns are
    dropped to avoid huge one-hot trees.
    """
    feat_cols: List[str] = []
    cat_cols: List[str] = []
    for c in df.columns:
        if c in _SKIP_FEATURE_COLS or c == cfg.target_col or c == cfg.time_col or c == cfg.key_col:
            continue
        if c.startswith("_"):
            continue
        s = df[c]
        if s.dtype == object or str(s.dtype) == "category":
            card = s.nunique(dropna=True)
            if card == 0:
                continue
            if card > max_cardinality_categorical:
                continue
            feat_cols.append(c)
            cat_cols.append(c)
        elif np.issubdtype(s.dtype, np.number) or np.issubdtype(s.dtype, np.bool_):
            feat_cols.append(c)
        # skip datetime and other dtypes
    # Preserve known categorical hints
    for c in list(feat_cols):
        if c in _CATEGORICAL_HINTS and c not in cat_cols:
            cat_cols.append(c)
    # Deduplicate while preserving order
    cat_cols = list(dict.fromkeys(cat_cols))
    feat_cols = list(dict.fromkeys(feat_cols))
    return feat_cols, cat_cols


def _prepare_x(
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    X = df.loc[:, feature_cols].copy()
    # JSON-native scalar types LightGBM's _json_default_with_numpy can
    # actually serialise.  Anything else (decimal.Decimal, pd.Timestamp,
    # custom classes, ...) is returned unchanged by that helper, which
    # makes json.dumps recurse on the same id and raise
    # "Circular reference detected" inside _dump_pandas_categorical.
    _JSON_NATIVE = (
        str, int, float, bool, type(None),
        np.integer, np.floating, np.bool_,
    )
    for c in categorical_cols:
        if c not in X.columns:
            continue
        # Drop any existing categorical wrapping so we always rebuild
        # from raw values (also resolves the legacy nested-categorical
        # case the previous fix targeted: a categorical whose
        # .categories is itself a CategoricalIndex).
        s_obj = X[c].astype("object")
        null_mask = X[c].isna()
        nn = s_obj[~null_mask]
        if len(nn) > 0 and not isinstance(nn.iloc[0], _JSON_NATIVE):
            # Non-JSON-native value detected (e.g. decimal.Decimal from
            # Spark→pandas conversion of DecimalType columns).  Stringify
            # the non-null values so LightGBM's
            #   json.dumps(pandas_categorical, default=_json_default_with_numpy)
            # in model_to_string() succeeds.  Without this the DMH path
            # raises ValueError("Circular reference detected") and the
            # inference dispatcher falls back to recursive forecasting,
            # adding ~80 minutes of wall-clock per run.
            #
            # `.where(null_mask, ...)` keeps original NaNs in place; only
            # non-null entries get stringified, so `np.nan` round-trips
            # cleanly through the subsequent astype("category").
            s_obj = s_obj.where(null_mask, s_obj.astype(str))
        X[c] = s_obj.astype("category")
    # Replace inf/-inf
    num_cols = [c for c in X.columns if c not in categorical_cols]
    if num_cols:
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    return X


# ---------------------------------------------------------------------------
# Build (X, y) for a single horizon h
# ---------------------------------------------------------------------------


def _build_horizon_panel(
    features_df: pd.DataFrame,
    horizon: int,
    cfg: DirectMHConfig,
) -> pd.DataFrame:
    """Attach the h-step-ahead target as column `_y_target` to features_df.

    Assumes features_df has one row per (key, year_week) and contains a
    target_col so that we can pull y[t+h] from other rows of the SAME key.

    **Performance**: avoids the per-horizon ``features_df.copy()`` and
    uses ``_yw_to_idx_vec`` instead of ``.apply(_yw_to_idx)``. If the
    caller already pre-computed ``_wk_idx`` on ``features_df`` (the
    train_direct_multihorizon hot path does this once outside the
    horizon loop), we reuse it instead of recomputing.

    Holds gaps correctly: the lookup is by ``(key, _wk_idx + h)`` so
    keys that skip weeks still resolve to the right ``y[t+h]`` when
    that target row exists, and to NaN otherwise.
    """
    key_col = cfg.key_col
    time_col = cfg.time_col
    target_col = cfg.target_col

    if time_col not in features_df.columns or target_col not in features_df.columns:
        raise ValueError(f"features_df must have columns {time_col} and {target_col}")

    # If `_wk_idx` was pre-computed by the caller, skip the (vectorised)
    # recomputation.  Otherwise vectorise it (50-200x faster than
    # `.apply(_yw_to_idx)` row-by-row).
    if "_wk_idx" in features_df.columns:
        df = features_df  # no copy — we add fresh columns below via merge,
                          # which returns a new frame anyway
        wk_idx = df["_wk_idx"].to_numpy()
    else:
        df = features_df.copy()
        wk_idx = _yw_to_idx_vec(df[time_col])
        df["_wk_idx"] = wk_idx

    target_idx = wk_idx + int(horizon)

    # Build a small (key, _wk_idx, target) lookup table and merge it
    # against the feature panel using `_target_idx`.  Pandas's hash join
    # on int columns is fast; the cost we used to pay was the Python-
    # level `.apply` populating `_wk_idx`, which we now skip.
    lookup_cols = [key_col, "_wk_idx", target_col]
    target_lookup = df.loc[:, lookup_cols].rename(
        columns={"_wk_idx": "_target_idx", target_col: "_y_target"}
    )
    out = df.assign(_target_idx=target_idx).merge(
        target_lookup, on=[key_col, "_target_idx"], how="left", copy=False
    )
    return out


# ---------------------------------------------------------------------------
# Train / predict primitives
# ---------------------------------------------------------------------------


def _parse_ensemble_spec(spec: str) -> List[Tuple[str, Optional[float], float]]:
    """Parse 'ensemble:regression=0.5,quantile@0.70=0.5' into
    [('regression', None, 0.5), ('quantile', 0.70, 0.5)].
    """
    if not spec.startswith("ensemble:"):
        return []
    body = spec[len("ensemble:"):]
    heads: List[Tuple[str, Optional[float], float]] = []
    total = 0.0
    for piece in body.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" in piece:
            name, w = piece.split("=", 1)
            w = float(w)
        else:
            name, w = piece, 1.0
        alpha = None
        if "@" in name:
            name, alpha_str = name.split("@", 1)
            alpha = float(alpha_str)
        heads.append((name.strip(), alpha, w))
        total += w
    if total > 0:
        heads = [(n, a, w / total) for (n, a, w) in heads]
    return heads


_AUTOLOG_DISABLED = False


def _belt_and_braces_disable_autolog() -> None:
    """Belt-and-braces guard so EVERY lgb.train inside DMH skips MLflow's
    autologger - even when the caller hasn't called the diq_runner top-level
    helper.  Idempotent + cheap (only runs the disable calls on the first
    call within this Python process).

    Also pre-sets env vars that disable CrewAI OTel tracing, LiteLLM
    telemetry, HuggingFace tokenizer parallelism, and Databricks-MLflow
    flavor-autoreg - these are the same bootstrap envs the diq_runner
    module-level block sets, repeated here so any caller that imports
    direct_multihorizon directly still gets them.
    """
    global _AUTOLOG_DISABLED
    if _AUTOLOG_DISABLED:
        return
    _AUTOLOG_DISABLED = True
    import os as _os
    if _os.environ.get("DIQ_RUNNER_KEEP_MLFLOW_AUTOLOG", "").lower() in (
        "1", "true", "yes",
    ):
        return
    for _k, _v in (
        ("MLFLOW_AUTOLOGGING_DISABLE",         "1"),
        ("MLFLOW_DATABRICKS_AUTOLOG_DISABLED", "true"),
        ("DISABLE_MLFLOW_INTEGRATION",         "TRUE"),
        ("OTEL_SDK_DISABLED",                  "true"),
        ("OTEL_TRACES_EXPORTER",               "none"),
        ("OTEL_METRICS_EXPORTER",              "none"),
        ("OTEL_LOGS_EXPORTER",                 "none"),
        ("CREWAI_TELEMETRY_OPT_OUT",           "true"),
        ("CREWAI_DISABLE_TELEMETRY",           "true"),
        ("CREWAI_DO_NOT_TRACK",                "1"),
        ("LITELLM_TELEMETRY",                  "False"),
        ("TOKENIZERS_PARALLELISM",             "false"),
        ("POSTHOG_DISABLED",                   "true"),
    ):
        _os.environ.setdefault(_k, _v)
    try:
        import mlflow
        try:
            mlflow.autolog(disable=True, silent=True)
        except Exception:
            pass
        for _flavor in ("lightgbm", "xgboost", "sklearn", "statsmodels",
                         "pytorch", "catboost", "tensorflow", "spark"):
            try:
                _f = getattr(mlflow, _flavor, None)
                if _f is not None:
                    _f.autolog(disable=True, silent=True)
            except Exception:
                pass
        try:
            from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS
            for _intg, _cfg in list(AUTOLOGGING_INTEGRATIONS.items()):
                if isinstance(_cfg, dict):
                    _cfg["disable"] = True
        except Exception:
            pass
    except ImportError:
        pass
    except Exception:
        pass


def _fit_lgbm_single(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    categorical_cols: List[str],
    cfg: DirectMHConfig,
    sample_weight_tr: Optional[np.ndarray] = None,
    sample_weight_val: Optional[np.ndarray] = None,
    objective_override: Optional[str] = None,
    alpha_override: Optional[float] = None,
):
    # Disable Databricks/MLflow's per-fit autologger before lightgbm gets
    # imported so the autolog hook never installs.  This is what was
    # causing the `🏃 View run burly-ant-45 at: ...` spam + 100x slowdown
    # in the user's notebook.
    _belt_and_braces_disable_autolog()
    import lightgbm as lgb

    obj = objective_override or cfg.objective
    # Thread count: default to all logical cores, but accept a config
    # override.  Setting this explicitly avoids LightGBM's auto-detect
    # being fooled by container cgroups on Databricks (which sometimes
    # reports 1 even when 16 cores are available).
    import os as _os
    n_threads = getattr(cfg, "num_threads", 0) or _os.cpu_count() or 1
    params = {
        "objective": obj,
        "metric": "mae",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l2": cfg.lambda_l2,
        "verbose": -1,
        "seed": cfg.random_state,
        # Thread + memory layout tuning.  `num_threads` pinned explicitly
        # so DBR's cgroup-confused auto-detect doesn't fall back to 1.
        # `force_col_wise=True` is the right choice for our shape (tall
        # panels with hundreds of features): col-wise histogram avoids
        # the row-wise heuristic's expensive sniff-test on every
        # lgb.train call AND uses less memory because it doesn't
        # materialise per-row caches.  This is THE recommended setting
        # for tabular forecasting workloads in the LightGBM docs.
        "num_threads": int(n_threads),
        "force_col_wise": True,
    }
    if obj == "tweedie":
        params["tweedie_variance_power"] = cfg.tweedie_variance_power
    if obj == "quantile":
        params["alpha"] = alpha_override if alpha_override is not None else 0.5
        params["metric"] = "quantile"

    lgb_train = lgb.Dataset(
        X_tr,
        y_tr,
        categorical_feature=categorical_cols or "auto",
        weight=sample_weight_tr,
    )
    lgb_val = lgb.Dataset(
        X_val,
        y_val,
        categorical_feature=categorical_cols or "auto",
        reference=lgb_train,
        weight=sample_weight_val,
    )
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=cfg.verbose_eval),
        ],
    )
    return model


def add_trajectory_features(df: pd.DataFrame, key_col: str = "key", target_col: str = "actual_sales") -> pd.DataFrame:
    """Per-key rolling trajectory features (slope, volatility, acceleration).

    Causally uses history up to each row's timestamp.  Adds:
        key_trend_4_13  : mean(last 4w) - mean(last 13w)   (short-term shift)
        key_trend_13_52 : mean(last 13w) - mean(last 52w)  (medium-term trend)
        key_accel       : difference of the two above
        key_recent_cv_13: std / |mean| over last 13w
        key_recent_over_long: mean(last 4w) / |mean(last 52w)|  (ratio to annual)

    These features specifically help close the WMAPE gap vs per-key benchmarks
    by injecting recent per-key dynamics into a pooled model.
    """
    if target_col not in df.columns:
        return df
    # `sort_values` already returns a new DataFrame, so the leading
    # `.copy()` was a redundant deep copy of a wide frame.  We still
    # `reset_index` so subsequent positional column assignments line up.
    df = df.sort_values([key_col, "year_week"]).reset_index(drop=True)
    g = df.groupby(key_col)[target_col]
    r4_mean = g.rolling(window=4, min_periods=2).mean().reset_index(0, drop=True)
    r13_mean = g.rolling(window=13, min_periods=3).mean().reset_index(0, drop=True)
    r52_mean = g.rolling(window=52, min_periods=10).mean().reset_index(0, drop=True)
    r13_std = g.rolling(window=13, min_periods=3).std().reset_index(0, drop=True)
    df["key_trend_4_13"] = r4_mean - r13_mean
    df["key_trend_13_52"] = r13_mean - r52_mean
    df["key_accel"] = df["key_trend_4_13"] - df["key_trend_13_52"]
    df["key_recent_cv_13"] = (r13_std / (r13_mean.abs() + 1e-3)).clip(0, 10)
    df["key_recent_over_long"] = (r4_mean / (r52_mean.abs() + 1e-3)).clip(0, 10)
    return df


def add_apg_sum_feature(
    df: pd.DataFrame,
    apg_col: str = "APG_code",
    target_col: str = "actual_sales",
) -> pd.DataFrame:
    """APG-level weekly sum, shifted by 1 week (no leak).

    Injects cross-sectional group signal that a single-key model cannot see.
    """
    if apg_col not in df.columns or target_col not in df.columns:
        return df
    # Don't deep-copy the whole frame just to coerce one column.
    # `assign` returns a new frame with the year_week dtype fixed; the
    # subsequent `merge` returns another fresh frame.  Net: one wide
    # copy avoided per call (was 2x).
    df = df.assign(**{"year_week": df["year_week"].astype(int)})
    agg = (
        df.groupby([apg_col, "year_week"])[target_col]
        .sum()
        .reset_index()
        .rename(columns={target_col: "apg_sum_prev"})
    )
    agg["year_week"] = agg["year_week"].astype(int) + 1  # shift by 1 week
    df = df.merge(agg, on=[apg_col, "year_week"], how="left")
    df["apg_sum_prev"] = df["apg_sum_prev"].fillna(0)
    return df


_FWD_CANDIDATE_COLS = [
    "week_of_year", "week_of_year_sin", "week_of_year_cos",
    "month", "month_sin", "month_cos", "quarter",
    "is_holiday", "holiday_flag", "weeks_to_nearest_holiday", "season",
    "promo_shipment_flag", "promo_num_days_shipped",
    "promo_cash_discount_per_case_off_invoice",
    "promo_instore_flag", "promo_num_days_instore",
    "promo_primary_mechanic", "promo_primary_objective", "promo_feature",
    "promo_instore_primary_mechanic", "promo_instore_feature",
    "WTHR_avgTemp", "WTHR_maxTemp", "WTHR_maxTempVsNorm", "high_sunshine",
]


def add_forward_target_features(
    df: pd.DataFrame,
    horizon: int,
    key_col: str = "key",
    candidate_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Attach target-week known features (calendar, promo, weather forecasts).

    For a row at origin t predicting y[t+h], this grabs columns from the row at
    (key, t+h) and prefixes them with ``fwd{h}_``.  Target-week promo/holiday
    flags are usually planned ahead in CPG so they're known at inference time.

    If ``candidate_cols`` is provided, it replaces the built-in UK-centric
    default list (``_FWD_CANDIDATE_COLS``) so TH/DE/etc. can supply their
    own calendar/promo/weather column names automatically derived by the
    inference helper from the config's feature declarations.
    """
    cols = list(candidate_cols) if candidate_cols else list(_FWD_CANDIDATE_COLS)
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    lk = df[[key_col, "year_week"] + present].copy()
    lk["year_week"] = lk["year_week"].astype(int)
    lk["_tw"] = lk["year_week"].apply(lambda x: yw_shift(int(x), horizon))
    rename_map = {c: f"fwd{horizon}_{c}" for c in present}
    lk = lk.rename(columns=rename_map)
    d = df.copy()
    d["year_week"] = d["year_week"].astype(int)
    d = d.merge(
        lk[[key_col, "_tw"] + [rename_map[c] for c in present]],
        left_on=[key_col, "year_week"],
        right_on=[key_col, "_tw"],
        how="left",
    ).drop(columns=["_tw"])
    return d


def augment_features(
    df: pd.DataFrame,
    horizon: int,
    key_col: str = "key",
    target_col: str = "actual_sales",
    apg_col: str = "APG_code",
    enable_trajectory: bool = True,
    enable_apg: bool = True,
    enable_forward: bool = True,
    forward_candidate_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compose the three augmenters proven to help on UK Unilever (iter 17-20).

    Caller typically applies once to train, once to val, once to the
    combined test panel so all frames share column schema.  All column
    names default to UK Unilever conventions but can be overridden for
    any other market:

    - ``target_col``  - TH uses "Actuals", UK uses "actual_sales"
    - ``apg_col``     - TH uses "APGDescription", UK uses "APG_code"
    - ``forward_candidate_cols`` - when None, falls back to the UK set;
      otherwise the inference helper passes the union of config-declared
      price/promo/holiday/weather feature names so each market picks up
      its own target-week known-ahead signals.
    """
    out = df
    if enable_trajectory:
        out = add_trajectory_features(out, key_col=key_col, target_col=target_col)
    if enable_apg:
        out = add_apg_sum_feature(out, apg_col=apg_col, target_col=target_col)
    if enable_forward:
        out = add_forward_target_features(
            out, horizon=horizon, key_col=key_col,
            candidate_cols=forward_candidate_cols,
        )
    return out


def _predict_one(model, X):
    """Predict with either a plain LightGBM Booster or an ensemble dict.

    If the training-time feature filter left a `_feature_cols` attribute on
    the model, we automatically subset X to those columns.
    """
    import numpy as _np
    if isinstance(model, dict) and model.get("_ensemble"):
        feat_cols = model.get("_feature_cols")
        X_use = X[feat_cols] if feat_cols else X
        agg = None
        for (_obj, _alpha, weight, m) in model["heads"]:
            p = m.predict(X_use, num_iteration=getattr(m, "best_iteration", None))
            agg = (p * weight) if agg is None else agg + p * weight
        return agg
    feat_cols = getattr(model, "_feature_cols", None)
    X_use = X[feat_cols] if feat_cols else X
    return model.predict(X_use, num_iteration=getattr(model, "best_iteration", None))


def _build_recency_weights(
    wk_idx: np.ndarray,
    halflife_weeks: float,
) -> Optional[np.ndarray]:
    """Return exponentially-decaying weights centred on the max week.

    halflife_weeks <= 0 disables (returns None).
    """
    if halflife_weeks is None or halflife_weeks <= 0:
        return None
    w = wk_idx.astype(float)
    max_w = w.max()
    # weight = 0.5 ** ((max_w - w) / halflife)
    exponent = (max_w - w) / halflife_weeks
    weights = np.power(0.5, exponent)
    # Ensure positive and finite
    weights = np.clip(weights, 1e-6, 1.0)
    return weights


# ---------------------------------------------------------------------------
# Public API: fit per-horizon models, predict for origin
# ---------------------------------------------------------------------------


@dataclass
class DirectMHArtifacts:
    """Container for all artifacts from direct multi-horizon training."""

    feature_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    models: Dict[int, object] = field(default_factory=dict)
    # Per-horizon, per-segment calibration factors.
    # Layout: {horizon: {segment: factor}}, with segment="__GLOBAL__" as fallback.
    calibration: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        import joblib

        for h, m in self.models.items():
            joblib.dump(m, os.path.join(path, f"direct_mh_model_h{h}.joblib"))
        meta = {
            "feature_cols": self.feature_cols,
            "categorical_cols": self.categorical_cols,
            "calibration": {str(k): v for k, v in self.calibration.items()},
            "horizons": sorted(self.models.keys()),
        }
        with open(os.path.join(path, "direct_mh_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DirectMHArtifacts":
        import joblib

        with open(os.path.join(path, "direct_mh_meta.json")) as f:
            meta = json.load(f)
        a = cls(
            feature_cols=meta["feature_cols"],
            categorical_cols=meta["categorical_cols"],
            calibration={int(k): v for k, v in meta.get("calibration", {}).items()},
        )
        for h in meta["horizons"]:
            a.models[int(h)] = joblib.load(os.path.join(path, f"direct_mh_model_h{h}.joblib"))
        return a


def train_direct_multihorizon(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    cfg: DirectMHConfig,
    key_to_segment: Optional[Dict[str, str]] = None,
    extra_features: Optional[pd.DataFrame] = None,
    backtest_origin: Optional[int] = None,
) -> DirectMHArtifacts:
    """Train one LightGBM per horizon on train (tuned on val).

    Both modes use the SAME target-observability filter
    ``origin + h <= effective_cutoff``, so the model only ever sees
    training rows whose label `y[t+h]` was actually observed in the
    data:

    - **Forward inference** (default, ``backtest_origin=None``):
      Combined panel = train + val (+ optional ``extra_features``).
      Effective cutoff = max year_week in the combined data
      (i.e. ``val_end`` - the production "now" boundary).
      Every origin from the start of history through ``val_end - h`` is
      a training example; the last ~13 origin-weeks before the cutoff
      become the early-stopping holdout.  This means val_features rows
      are USED FOR TRAINING (with appropriate horizon-aware filtering),
      not just early stopping.

    - **Rolling-origin backtest** (``backtest_origin`` set to the
      origin's train_cutoff as YYYYWW int):
      Combined panel = train + val + ``extra_features`` (typically
      test_features, so late origins can train on early test weeks that
      would legitimately be observed by that point in time).  Effective
      cutoff = backtest_origin.  Same per-horizon filter, same last-13
      holdout - so each origin retrains using exactly the data it would
      have access to, never a week it wouldn't.  Proper rolling-origin
      semantics, no leak.
    """
    logger.info("=" * 60)
    logger.info("DIRECT MULTI-HORIZON TRAINING")
    logger.info("=" * 60)
    extra_n = 0 if extra_features is None else len(extra_features)
    logger.info(
        "train rows=%s val rows=%s extra rows=%s horizons=%s objective=%s backtest_origin=%s",
        len(train_features),
        len(val_features),
        extra_n,
        list(cfg.horizons),
        cfg.objective,
        backtest_origin,
    )
    t_start = time.time()

    # Combine panels for horizon-target lookup.
    frames = [train_features, val_features]
    if extra_features is not None and len(extra_features) > 0:
        frames.append(extra_features)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values([cfg.key_col, cfg.time_col]).reset_index(drop=True)

    # Pre-compute `_wk_idx` ONCE on the combined panel.  Without this
    # `_build_horizon_panel` re-runs `df[time_col].apply(_yw_to_idx)` on
    # every horizon (13x by default), and `.apply` with a Python
    # function is by far the slowest pattern in pandas.  Doing it once
    # vectorised cuts the panel-build phase from O(13 * apply) to
    # O(1 * vec).
    combined["_wk_idx"] = _yw_to_idx_vec(combined[cfg.time_col])

    # Pick feature list once (on combined so dtypes are consistent).
    feature_cols, categorical_cols = _select_feature_columns(combined, cfg)
    logger.info(
        "Selected %s features (%s categorical)", len(feature_cols), len(categorical_cols)
    )

    artifacts = DirectMHArtifacts(
        feature_cols=feature_cols, categorical_cols=categorical_cols
    )

    # Boundary for forward-inference mode: last week of the original train set.
    train_end_yw = int(train_features[cfg.time_col].max())

    # Global calibration by horizon & segment
    cal_by_horizon: Dict[int, Dict[str, float]] = {}

    # ════════════════════════════════════════════════════════════════════
    # DRIVER-SIDE HORIZON PARALLELISM
    # ════════════════════════════════════════════════════════════════════
    # Historically this was `for h in cfg.horizons:` — i.e. H × n_seeds ×
    # n_heads LightGBM fits back-to-back on the driver process while the
    # Spark workers idled.  For a typical 'auto' configuration that's
    # 13 horizons × 3 heads × 3 seeds = 117 serial fits, which dominates
    # DIQ wall time (and explains why a single category run could exceed
    # 5 hours on a cluster that had 64 idle worker cores).
    #
    # Every horizon is mathematically independent of every other horizon:
    #   * its own _y_target column (lag-h shift)
    #   * its own target-observability filter (origin + h <= cutoff)
    #   * its own top-k feature probe
    #   * its own ensemble fit + auto-tune
    # The only shared state is the read-only `combined` panel, which can
    # be safely accessed by multiple threads.  LightGBM releases the GIL
    # during training, so a ThreadPoolExecutor produces real parallelism
    # without needing process-level forking (which would otherwise copy
    # the multi-GB combined frame N times).
    #
    # Two safety guarantees:
    # 1. Numerical identity — each horizon produces the exact same model
    #    it would have under the old serial loop.  All inputs (data,
    #    seed, cfg) are pinned per-horizon; only the dispatch order
    #    changes.
    # 2. No OMP oversubscription — we shrink per-fit LightGBM `num_threads`
    #    proportionally so `horizon_workers * num_threads ≈ driver cores`.
    #    Without this, 4 horizons × all-cores-each would thrash the CPU.
    # ════════════════════════════════════════════════════════════════════
    import os as _os_dmh
    import concurrent.futures as _futures_dmh

    _cpu_avail = _os_dmh.cpu_count() or 1
    _requested_hw = int(getattr(cfg, "horizon_workers", 0) or 0)
    if _requested_hw <= 0:
        # Auto: half the driver cores, capped at the horizon count.  Half
        # rather than full so each LightGBM fit still gets >=2 threads
        # (force_col_wise + num_threads=1 is unusably slow on wide panels).
        _horizon_workers = max(1, min(len(cfg.horizons), max(1, _cpu_avail // 2)))
    else:
        _horizon_workers = max(1, min(len(cfg.horizons), _requested_hw))

    # Per-fit LightGBM thread budget.  `cfg.num_threads` is the
    # caller-requested cap (0 = use os.cpu_count()); we divide that by
    # the worker count and pass the result down through `_cfg_worker`.
    _orig_num_threads = int(getattr(cfg, "num_threads", 0) or 0) or _cpu_avail
    _per_fit_threads = max(1, _orig_num_threads // _horizon_workers)

    if _horizon_workers > 1:
        # Build a worker-config with throttled num_threads.  Every
        # `_fit_lgbm_single` and `_fit_one` call inside the helper reads
        # `cfg_w.num_threads`, so this single override propagates to
        # every LightGBM fit on every horizon without a wider refactor.
        _cfg_worker = DirectMHConfig(
            **{**cfg.__dict__, "num_threads": _per_fit_threads}
        )
        logger.info(
            "DMH horizon dispatch: ThreadPool max_workers=%d, "
            "per-fit LightGBM threads=%d (driver cores=%d, horizons=%d)",
            _horizon_workers, _per_fit_threads, _cpu_avail, len(cfg.horizons),
        )
    else:
        # Serial path — identical to pre-parallelisation behaviour.
        _cfg_worker = cfg
        logger.info(
            "DMH horizon dispatch: serial (1 worker, %d horizons)",
            len(cfg.horizons),
        )

    def _train_one_horizon(h):
        """Train DMH heads for a single horizon.

        Closure-captured: ``combined``, ``_cfg_worker``, ``feature_cols``,
        ``categorical_cols``, ``backtest_origin``, ``key_to_segment``.

        Returns a dict consumed by the dispatcher to populate
        ``artifacts.models[h]`` and ``cal_by_horizon[h]``, or ``None``
        when the horizon has no usable training rows.
        """
        t_h = time.time()
        cfg_w = _cfg_worker  # alias so the inner body reads cleanly
        panel = _build_horizon_panel(combined, h, cfg_w)
        # Drop rows where the h-step target is missing.
        panel = panel.dropna(subset=["_y_target"]).reset_index(drop=True)
        if panel.empty:
            logger.warning("Horizon %s: no training rows - skipping", h)
            return None

        origin_yw = panel[cfg_w.time_col].astype(int)

        # ----------------------------------------------------------------
        # Decide the cutoff for what counts as "training data with an
        # observable target":
        #   - backtest_origin set  -> the rolling-origin cutoff
        #   - forward inference    -> the latest period in the input data
        #     (i.e. val_end - the production "now" boundary).  Using
        #     train_end_yw here as we used to was leaving every val-period
        #     observation on the floor; for a forward forecast we want
        #     EVERY origin whose h-step target was actually observed.
        # ----------------------------------------------------------------
        if backtest_origin is not None:
            effective_cutoff = int(backtest_origin)
            cutoff_label = f"backtest_origin={backtest_origin}"
        else:
            effective_cutoff = int(combined[cfg_w.time_col].astype(int).max())
            cutoff_label = f"forward (val_end={effective_cutoff})"

        # Per-horizon target observability filter: origin + h <= cutoff,
        # i.e. the target label `y[t+h]` was actually observed in the data
        # we're training on.  Anything beyond that would be a leak.
        #
        # Vectorised: equivalent to comparing `_yw_to_idx(o) + h` against
        # `_yw_to_idx(cutoff)` directly.  Avoids the per-row
        # `origin_yw.apply(yw_shift)` Python loop (was 1M+ Python calls
        # per horizon on a typical UK frame).
        cutoff_idx = _yw_to_idx(int(effective_cutoff))
        origin_idx = panel["_wk_idx"].to_numpy() if "_wk_idx" in panel.columns else _yw_to_idx_vec(origin_yw)
        target_observable = (origin_idx + int(h)) <= cutoff_idx
        panel = panel.loc[target_observable].reset_index(drop=True)
        if panel.empty:
            logger.warning(
                "Horizon %s: no rows usable at %s - skipping", h, cutoff_label,
            )
            return None
        origin_yw = panel[cfg_w.time_col].astype(int)

        # Split the eligible origins into train / val for early stopping.
        # The last ~13 origin-weeks (or 20 % of unique origins, whichever
        # is smaller) are held out so LightGBM has a clean ES signal that
        # mimics the production forecast horizon - critically WITHOUT
        # discarding all of val_features as wholly-untrained-on the way
        # the previous forward-inference branch did.
        unique_origins = sorted(origin_yw.unique())
        val_origin_count = max(1, min(13, int(len(unique_origins) * 0.2)))
        val_origins = set(unique_origins[-val_origin_count:])
        is_val = origin_yw.isin(val_origins)
        train_panel = panel.loc[~is_val].reset_index(drop=True)
        val_panel = panel.loc[is_val].reset_index(drop=True)
        logger.info(
            "  h=%d: train_origins=%d, val_origins=%d (cutoff: %s)",
            int(h), len(train_panel), len(val_panel), cutoff_label,
        )

        if train_panel.empty or val_panel.empty:
            # Fall back to time-ordered 95/5 split so long horizons (where
            # there's no "val" target yet) still train.
            logger.warning(
                "Horizon %s: empty split (tr=%s, val=%s) - falling back to last 5%% of training as val"
                % (h, len(train_panel), len(val_panel))
            )
            if panel.empty:
                return None
            panel_sorted = panel.sort_values(cfg_w.time_col).reset_index(drop=True)
            split_idx = max(1, int(len(panel_sorted) * 0.95))
            train_panel = panel_sorted.iloc[:split_idx].reset_index(drop=True)
            val_panel = panel_sorted.iloc[split_idx:].reset_index(drop=True)
            if train_panel.empty or val_panel.empty:
                logger.warning("Horizon %s: fallback also produced empty split - skipping", h)
                return None

        X_tr = _prepare_x(train_panel, feature_cols, categorical_cols)
        y_tr = train_panel["_y_target"].values.astype(float)
        X_val = _prepare_x(val_panel, feature_cols, categorical_cols)
        y_val = val_panel["_y_target"].values.astype(float)

        if cfg_w.use_log_target:
            y_tr = np.log1p(np.maximum(0, y_tr))
            y_val_transformed = np.log1p(np.maximum(0, y_val))
        else:
            y_val_transformed = y_val

        # Recency weights - emphasise recent origins to combat distribution shift.
        tr_wk_idx = train_panel[cfg_w.time_col].apply(_yw_to_idx).values
        val_wk_idx = val_panel[cfg_w.time_col].apply(_yw_to_idx).values
        w_tr = _build_recency_weights(tr_wk_idx, cfg_w.recency_halflife_weeks)
        w_val = _build_recency_weights(val_wk_idx, cfg_w.recency_halflife_weeks)

        # Feature-importance filter: do a cheap probe fit, take top_k by gain.
        horizon_feature_cols = list(feature_cols)
        horizon_categorical_cols = list(categorical_cols)
        if cfg_w.top_k_features and len(feature_cols) > cfg_w.top_k_features:
            import lightgbm as _lgb_probe
            import os as _os
            _probe_threads = getattr(cfg_w, "num_threads", 0) or _os.cpu_count() or 1
            probe_params = {
                "objective": "regression",
                "metric": "mae",
                "learning_rate": 0.1,
                "num_leaves": 31,
                "min_data_in_leaf": cfg_w.min_data_in_leaf,
                "verbose": -1,
                "seed": cfg_w.random_state,
                # Same thread/memory layout tuning as the main fit
                "num_threads": int(_probe_threads),
                "force_col_wise": True,
            }
            dtr_probe = _lgb_probe.Dataset(X_tr, y_tr, categorical_feature=categorical_cols or "auto")
            dval_probe = _lgb_probe.Dataset(X_val, y_val_transformed, categorical_feature=categorical_cols or "auto", reference=dtr_probe)
            probe = _lgb_probe.train(
                probe_params,
                dtr_probe,
                num_boost_round=cfg_w.top_k_filter_iterations,
                valid_sets=[dval_probe],
                callbacks=[_lgb_probe.early_stopping(30, verbose=False), _lgb_probe.log_evaluation(period=-1)],
            )
            importance = pd.Series(probe.feature_importance(importance_type="gain"), index=X_tr.columns)
            top_features = importance.sort_values(ascending=False).head(cfg_w.top_k_features).index.tolist()
            horizon_feature_cols = top_features
            horizon_categorical_cols = [c for c in categorical_cols if c in top_features]
            X_tr = X_tr[top_features]
            X_val = X_val[top_features]
            logger.info(
                "  h=%d: filtered %d -> %d features (importance probe)"
                % (int(h), len(feature_cols), len(top_features))
            )

        def _fit_one(obj_override, alpha_override, seed_offset):
            cfg_seed = DirectMHConfig(**{**cfg_w.__dict__})
            cfg_seed.random_state = cfg_w.random_state + seed_offset
            return _fit_lgbm_single(
                X_tr,
                y_tr,
                X_val,
                y_val_transformed,
                horizon_categorical_cols,
                cfg_seed,
                sample_weight_tr=w_tr,
                sample_weight_val=w_val,
                objective_override=obj_override,
                alpha_override=alpha_override,
            )

        ensemble_spec = _parse_ensemble_spec(cfg_w.objective)
        auto_mode = str(cfg_w.objective).strip().lower() == "auto"

        if auto_mode:
            # Train all candidate heads, then learn ensemble weights from val OOF.
            head_models: List[Tuple[str, Optional[float], float, object]] = []
            head_val_preds: List[np.ndarray] = []
            head_labels: List[Tuple[str, Optional[float]]] = []
            for head_idx, (head_obj, head_alpha) in enumerate(cfg_w.auto_tune_candidates):
                # Bag each head across seeds, average seeds into a single head prediction.
                seed_models = []
                seed_val_preds = []
                for s in range(max(1, cfg_w.n_seeds)):
                    m = _fit_one(head_obj, head_alpha, seed_offset=head_idx * 7 + s * cfg_w.seed_bag_step)
                    seed_models.append(m)
                    vp = m.predict(X_val, num_iteration=getattr(m, "best_iteration", None))
                    if cfg_w.use_log_target:
                        vp = np.expm1(vp)
                    seed_val_preds.append(np.clip(vp, 0, None))
                # Store bagged model as a mini-ensemble; its head weight is learned below.
                for _s_idx, sm in enumerate(seed_models):
                    head_models.append((head_obj, head_alpha, 1.0 / len(seed_models), sm))
                head_val_preds.append(np.mean(seed_val_preds, axis=0))
                head_labels.append((head_obj, head_alpha))

            # Optionally subset val to the most-recent K weeks for a tuning
            # signal closer to the test regime.
            y_val_for_tune = y_val
            head_val_preds_for_tune = head_val_preds
            recent_k = int(cfg_w.auto_tune_recent_weeks or 0)
            if recent_k > 0 and len(val_panel) > 0:
                vw = val_panel[cfg_w.time_col].astype(int).values
                recent_threshold = np.sort(np.unique(vw))[max(0, len(np.unique(vw)) - recent_k)]
                recent_mask = vw >= recent_threshold
                if recent_mask.sum() > 50:
                    y_val_for_tune = y_val[recent_mask]
                    head_val_preds_for_tune = [p[recent_mask] for p in head_val_preds]
                    logger.info(
                        "  h=%d: auto-tune using last %d val weeks (%d rows)"
                        % (int(h), recent_k, int(recent_mask.sum()))
                    )

            # Learn optimal weights on val OOF.
            learned_weights = _learn_ensemble_weights(
                head_val_preds=head_val_preds_for_tune,
                y_val=y_val_for_tune,
                bias_weight=cfg_w.auto_tune_bias_weight,
                grid_step=cfg_w.auto_tune_grid_step,
            )

            # Shrinkage toward uniform to hedge against val→test distribution shift.
            s = float(cfg_w.auto_tune_shrinkage_to_uniform)
            if s > 0:
                n_heads = len(learned_weights)
                uniform_w = np.full(n_heads, 1.0 / n_heads)
                learned_weights = (1 - s) * learned_weights + s * uniform_w
                learned_weights = learned_weights / max(learned_weights.sum(), 1e-9)
            logger.info(
                "  h=%d: auto-tuned ensemble weights -> %s"
                % (int(h), ", ".join([f"{lbl[0]}{'@'+str(lbl[1]) if lbl[1] else ''}={w:.2f}" for lbl, w in zip(head_labels, learned_weights)]))
            )

            # Rewrite head weights so predict-time uses learned weights per head (split across seeds).
            rewritten_heads = []
            n_seeds_per_head = max(1, cfg_w.n_seeds)
            for hi, (lbl, lw) in enumerate(zip(head_labels, learned_weights)):
                # heads for this head-index span seeds
                for si in range(n_seeds_per_head):
                    idx = hi * n_seeds_per_head + si
                    head_obj, head_alpha, _old_w, m = head_models[idx]
                    rewritten_heads.append((head_obj, head_alpha, float(lw) / n_seeds_per_head, m))
            model = {"_ensemble": True, "heads": rewritten_heads,
                     "_auto_tune_info": {"head_labels": head_labels,
                                          "learned_weights": [float(w) for w in learned_weights]}}
        elif ensemble_spec:
            head_models = []
            for head_idx, (head_obj, head_alpha, head_weight) in enumerate(ensemble_spec):
                for s in range(max(1, cfg_w.n_seeds)):
                    m = _fit_one(head_obj, head_alpha, seed_offset=head_idx * 7 + s * cfg_w.seed_bag_step)
                    # Each bagged model within a head gets weight/n_seeds.
                    head_models.append((head_obj, head_alpha, head_weight / max(1, cfg_w.n_seeds), m))
            model = {"_ensemble": True, "heads": head_models}
        else:
            if cfg_w.n_seeds > 1:
                head_models = []
                for s in range(cfg_w.n_seeds):
                    m = _fit_one(None, None, seed_offset=s * cfg_w.seed_bag_step)
                    head_models.append((cfg_w.objective, None, 1.0 / cfg_w.n_seeds, m))
                model = {"_ensemble": True, "heads": head_models}
            else:
                model = _fit_one(None, None, seed_offset=0)
        # Remember the (possibly filtered) feature list for prediction.
        if cfg_w.top_k_features:
            # For ensemble models we stash the per-horizon feature list on the dict; for
            # single models, we store on the object by attaching an attribute.
            if isinstance(model, dict):
                model["_feature_cols"] = horizon_feature_cols
                model["_categorical_cols"] = horizon_categorical_cols
            else:
                setattr(model, "_feature_cols", horizon_feature_cols)
                setattr(model, "_categorical_cols", horizon_categorical_cols)

        # Compute val predictions for calibration (supports ensemble or single model).
        if isinstance(model, dict) and model.get("_ensemble"):
            agg = None
            for (ho, ha, hw, mm) in model["heads"]:
                p = mm.predict(X_val, num_iteration=getattr(mm, "best_iteration", None))
                agg = (p * hw) if agg is None else agg + p * hw
            val_pred_raw = agg
        else:
            val_pred_raw = model.predict(
                X_val, num_iteration=getattr(model, "best_iteration", None)
            )
        if cfg_w.use_log_target:
            val_pred = np.expm1(val_pred_raw)
        else:
            val_pred = val_pred_raw
        val_pred = np.maximum(0.0, val_pred)

        # Learn per-segment calibration on validation.
        cal_dict = _learn_segment_calibration(
            val_pred,
            y_val,
            val_panel[cfg_w.key_col].astype(str).values,
            key_to_segment or {},
            cfg_w,
        )

        wmape_val = _wmape(y_val, val_pred)
        bias_val = _bias(y_val, val_pred)
        # Per-horizon summary log emitted inline so the metric is
        # visible whether dispatch was serial or parallel.  Python's
        # logging module is thread-safe; the order across horizons may
        # interleave under ThreadPool but each line is intact.
        logger.info(
            "  h=%d | train_n=%d val_n=%d best_iter=%s | val WMAPE=%.3f bias=%+.3f | cal_n=%d elapsed=%.1fs"
            % (
                int(h),
                int(len(X_tr)),
                int(len(X_val)),
                str(getattr(model, "best_iteration", -1)),
                float(wmape_val),
                float(bias_val),
                int(len(cal_dict)),
                float(time.time() - t_h),
            )
        )

        return {"h": h, "model": model, "cal_dict": cal_dict}

    def _apply_horizon_result(res):
        """Merge a `_train_one_horizon` result into outer-scope state.

        Always called on the main thread (dispatcher loop), so no
        synchronisation needed on `artifacts.models` / `cal_by_horizon`.
        """
        if res is None:
            return
        artifacts.models[res["h"]] = res["model"]
        cal_by_horizon[res["h"]] = res["cal_dict"]

    if _horizon_workers == 1:
        # Serial dispatch — identical control flow to the pre-parallel
        # version.  Kept as a separate branch so single-worker runs pay
        # zero ThreadPool/futures overhead.
        for h in cfg.horizons:
            _apply_horizon_result(_train_one_horizon(h))
    else:
        with _futures_dmh.ThreadPoolExecutor(max_workers=_horizon_workers) as _ex:
            _futs = {_ex.submit(_train_one_horizon, h): h for h in cfg.horizons}
            for _fut in _futures_dmh.as_completed(_futs):
                _h_for_fut = _futs[_fut]
                try:
                    _apply_horizon_result(_fut.result())
                except Exception as _exc:
                    # Don't crash the whole DMH run if one horizon
                    # blows up — log + skip, the dispatcher will simply
                    # leave that horizon out of artifacts.models, and
                    # downstream prediction handles missing horizons.
                    logger.exception(
                        "DMH horizon %s training failed: %s",
                        _h_for_fut, _exc,
                    )

    artifacts.calibration = cal_by_horizon
    logger.info("DIRECT MH training done: total %.1fs", time.time() - t_start)
    return artifacts


def _learn_ensemble_weights(
    head_val_preds: List[np.ndarray],
    y_val: np.ndarray,
    bias_weight: float = 1.0,
    grid_step: float = 0.1,
) -> np.ndarray:
    """Learn non-negative simplex weights over N head predictions.

    Minimises: WMAPE(blend, y_val) + bias_weight * |Bias(blend, y_val)|
    subject to sum(w) = 1, w >= 0.

    Strategy:
      1. Two-stage grid search (coarse step=grid_step, then refine by /5 near best).
      2. SLSQP refinement from grid winner if scipy is available.

    The composite loss balances accuracy (WMAPE) and calibration (bias) in a
    single scalar so we don't need per-category hand-weighting.
    """
    n = len(head_val_preds)
    if n == 0:
        return np.array([], dtype=float)
    stacked = np.stack(head_val_preds, axis=0)  # (n_heads, n_samples)
    y_sum = float(np.sum(np.abs(y_val)))
    if y_sum < 1e-9:
        return np.full(n, 1.0 / n)

    def loss_fn(w: np.ndarray) -> float:
        w = np.clip(w, 0, None)
        s = w.sum()
        if s < 1e-12:
            return 1e9
        w = w / s
        blend = (stacked * w[:, None]).sum(axis=0)
        wmape = float(np.sum(np.abs(blend - y_val)) / y_sum)
        bias_pct = float(np.sum(blend - y_val) / y_sum)
        return wmape + bias_weight * abs(bias_pct)

    # Coarse grid on simplex (compositions summing to 1 with step `grid_step`).
    step = float(grid_step)
    levels = int(round(1.0 / step))

    def _simplex_iter(remaining_levels: int, depth: int):
        if depth == n - 1:
            yield (remaining_levels,)
            return
        for k in range(0, remaining_levels + 1):
            for rest in _simplex_iter(remaining_levels - k, depth + 1):
                yield (k,) + rest

    best_loss = float("inf")
    best_w = None
    # Cap combinatoric explosion by dropping to a smaller `levels` when n is big.
    while _choose_ok(levels, n) is False:
        levels = max(4, levels - 1)
    for combo in _simplex_iter(levels, 0):
        w = np.array(combo, dtype=float) / levels
        L = loss_fn(w)
        if L < best_loss:
            best_loss = L
            best_w = w

    # Optional refinement with SLSQP.
    try:
        from scipy.optimize import minimize
        def loss_refine(w_free):
            w = np.concatenate([w_free, [1.0 - w_free.sum()]])
            return loss_fn(w)
        x0 = best_w[:-1].copy() if best_w is not None else np.full(n - 1, 1.0 / n)
        bounds = [(0.0, 1.0)] * (n - 1)
        cons = ({"type": "ineq", "fun": lambda x: 1.0 - x.sum()},)
        res = minimize(loss_refine, x0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 50, "ftol": 1e-5})
        if res.success:
            w_free = np.clip(res.x, 0, 1)
            w_last = max(0.0, 1.0 - w_free.sum())
            w_refined = np.concatenate([w_free, [w_last]])
            w_refined = w_refined / max(w_refined.sum(), 1e-9)
            L_refined = loss_fn(w_refined)
            if L_refined < best_loss:
                best_w = w_refined
                best_loss = L_refined
    except Exception:
        pass

    if best_w is None:
        best_w = np.full(n, 1.0 / n)
    return best_w / max(best_w.sum(), 1e-9)


def _choose_ok(levels: int, n_heads: int, cap: int = 5000) -> bool:
    """Cheap guard: simplex grid size <= cap."""
    from math import comb
    try:
        return comb(levels + n_heads - 1, n_heads - 1) <= cap
    except Exception:
        return False


def _learn_segment_calibration(
    preds: np.ndarray,
    actuals: np.ndarray,
    keys: np.ndarray,
    key_to_segment: Dict[str, str],
    cfg: DirectMHConfig,
) -> Dict[str, float]:
    """Learn multiplicative correction factor per segment (+ global fallback)."""
    if not cfg.enable_bias_calibration:
        return {}
    df = pd.DataFrame(
        {
            "key": keys,
            "pred": preds,
            "actual": actuals,
        }
    )
    df["segment"] = df["key"].map(key_to_segment).fillna("__DEFAULT__")

    factors: Dict[str, float] = {}
    # Per-segment
    for seg, grp in df.groupby("segment"):
        p = grp["pred"].sum()
        a = grp["actual"].sum()
        if len(grp) < cfg.calibration_min_samples or p <= 0 or a <= 0:
            continue
        factor = float(np.clip(a / p, cfg.calibration_min, cfg.calibration_max))
        factors[str(seg)] = factor
    # Global fallback
    p = df["pred"].sum()
    a = df["actual"].sum()
    if p > 0 and a > 0:
        factors["__GLOBAL__"] = float(
            np.clip(a / p, cfg.calibration_min, cfg.calibration_max)
        )
    else:
        factors["__GLOBAL__"] = 1.0
    return factors


def predict_rolling_origin(
    all_features: pd.DataFrame,
    artifacts: DirectMHArtifacts,
    cfg: DirectMHConfig,
    target_weeks: Sequence[int],
    horizon: int,
    key_to_segment: Optional[Dict[str, str]] = None,
    apply_calibration: bool = True,
) -> pd.DataFrame:
    """Rolling-origin forecasts at a fixed lag/horizon.

    For each target_week in target_weeks, find feature rows at
    origin = target_week - horizon (in week units), run them through the
    horizon-`h` model, and return (key, target_week, predicted).  This is how
    PristineTVF-style lag-4 backtests are produced.
    """
    if horizon not in artifacts.models:
        raise ValueError(
            f"No model for horizon={horizon}; available={sorted(artifacts.models.keys())}"
        )
    model = artifacts.models[horizon]
    parts: List[pd.DataFrame] = []
    for tw in target_weeks:
        origin_wk = yw_shift(tw, -horizon)
        df = all_features.loc[
            all_features[cfg.time_col].astype(int) == int(origin_wk)
        ].copy()
        if df.empty:
            logger.info("predict_rolling_origin: no origin rows at %s", origin_wk)
            continue
        X = _prepare_x(df, artifacts.feature_cols, artifacts.categorical_cols)
        pred = _predict_one(model, X)
        if cfg.use_log_target:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        out = pd.DataFrame(
            {
                cfg.key_col: df[cfg.key_col].values,
                "target_week": int(tw),
                "year_week": int(tw),
                "origin_week": int(origin_wk),
                "horizon": int(horizon),
                "lag": int(horizon),
                "predicted_raw": pred,
            }
        )
        parts.append(out)
    if not parts:
        return pd.DataFrame(
            columns=[cfg.key_col, "target_week", "year_week", "origin_week", "horizon", "lag", "predicted_raw", "predicted", "calibration_factor"]
        )
    forecasts = pd.concat(parts, ignore_index=True)
    forecasts["predicted"] = forecasts["predicted_raw"]
    forecasts["calibration_factor"] = 1.0

    if apply_calibration and artifacts.calibration:
        segs = (
            forecasts[cfg.key_col]
            .astype(str)
            .map(key_to_segment or {})
            .fillna("__DEFAULT__")
            .astype(str)
            .values
        )
        fmap = artifacts.calibration.get(int(horizon), {})
        default = fmap.get("__GLOBAL__", 1.0)
        factors = np.array([fmap.get(s, default) for s in segs], dtype=float)
        forecasts["calibration_factor"] = factors
        forecasts["predicted"] = np.clip(forecasts["predicted_raw"].values * factors, 0, None)

    return forecasts


def predict_direct_multihorizon(
    test_features: pd.DataFrame,
    artifacts: DirectMHArtifacts,
    cfg: DirectMHConfig,
    origin_week: int,
    key_to_segment: Optional[Dict[str, str]] = None,
    apply_calibration: bool = True,
    fallback_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Produce per-(key, target_week) forecasts from origin_week.

    `test_features` should contain one row per key at time == origin_week.  If
    the caller passes a wider panel we take rows with `year_week == origin_week`.
    If test_features does not contain origin_week (e.g. origin is the val_end
    week which lives in val_features), pass `fallback_features` (typically
    val_features) and the function searches there first.
    """
    if artifacts.feature_cols is None or not artifacts.models:
        raise ValueError("DirectMHArtifacts is empty - train first")

    df = test_features.loc[test_features[cfg.time_col].astype(int) == int(origin_week)].copy()
    if df.empty and fallback_features is not None:
        df = fallback_features.loc[
            fallback_features[cfg.time_col].astype(int) == int(origin_week)
        ].copy()
        if not df.empty:
            logger.info(
                "Origin week %s taken from fallback_features (%s rows)",
                origin_week,
                len(df),
            )
    if df.empty:
        # Final fallback: take last row per key from whichever frame is larger
        source = test_features if len(test_features) >= (
            0 if fallback_features is None else len(fallback_features)
        ) else fallback_features
        df = (
            source.sort_values([cfg.key_col, cfg.time_col])
            .groupby(cfg.key_col, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        logger.warning(
            "No rows at year_week=%s in features; falling back to last-per-key (%s rows)",
            origin_week,
            len(df),
        )

    # Prepare X once - it's the same set of features used at every horizon head.
    X = _prepare_x(df, artifacts.feature_cols, artifacts.categorical_cols)

    # SPLY lookup table: for each (key, target_week) we may want the lag-52
    # value.  We use the feature column configured in cfg.sply_col; when that
    # column is available on the origin row it already represents the lag-52
    # value ending at the ORIGIN week - but we want lag-52 relative to the
    # TARGET week.  To avoid needing the full historical panel at predict time
    # we approximate SPLY by the origin-row lag_52 adjusted by horizon offset
    # through the actual_sales_lag_{h} columns when they exist.  For a clean
    # blend we just use the `sply_col` value as a conservative seasonal
    # reference - this is robust and generalises.
    sply_vals: Optional[np.ndarray] = None
    if cfg.sply_blend_alpha < 1.0:
        for col in cfg.sply_candidate_cols:
            if col in df.columns:
                raw = np.asarray(df[col].fillna(0).values, dtype=float)
                if "log" in col:
                    raw = np.expm1(np.clip(raw, 0, 30))  # safeguard against huge values
                sply_vals = np.clip(raw, 0, None)
                logger.info("SPLY blend using column '%s' (alpha=%.2f)", col, cfg.sply_blend_alpha)
                break

    # Pre-compute predictions at every available horizon once.
    preds_by_h: Dict[int, np.ndarray] = {}
    for h, model in artifacts.models.items():
        p = _predict_one(model, X)
        if cfg.use_log_target:
            p = np.expm1(p)
        preds_by_h[int(h)] = np.clip(p, 0, None)

    def _smoothed(h_target: int) -> np.ndarray:
        """Apply horizon smoothing if configured; otherwise return raw h prediction."""
        base = preds_by_h.get(int(h_target))
        if cfg.smooth_window <= 0 or base is None:
            return base
        lo = int(h_target) - int(cfg.smooth_window)
        hi = int(h_target) + int(cfg.smooth_window)
        neighbours = [preds_by_h[h_] for h_ in range(lo, hi + 1) if h_ in preds_by_h]
        if not neighbours:
            return base
        if cfg.smooth_weights and len(cfg.smooth_weights) == len(neighbours):
            wts = np.asarray(cfg.smooth_weights, dtype=float)
            wts = wts / wts.sum()
            stacked = np.stack(neighbours, axis=0)
            return (stacked * wts[:, None]).sum(axis=0)
        return np.mean(neighbours, axis=0)

    parts: List[pd.DataFrame] = []
    for h in sorted(preds_by_h.keys()):
        pred = _smoothed(h)
        if sply_vals is not None:
            alpha = float(cfg.sply_blend_alpha)
            pred = alpha * pred + (1 - alpha) * sply_vals

        target_week_int = yw_shift(origin_week, h)
        out = pd.DataFrame(
            {
                cfg.key_col: df[cfg.key_col].values,
                "target_week": target_week_int,
                "year_week": target_week_int,
                "origin_week": origin_week,
                "horizon": h,
                "lag": h,
                "predicted_raw": pred,
            }
        )
        parts.append(out)

    forecasts = pd.concat(parts, ignore_index=True)

    # Apply per-horizon, per-segment calibration.
    if apply_calibration and artifacts.calibration:
        segs = (
            forecasts[cfg.key_col]
            .astype(str)
            .map(key_to_segment or {})
            .fillna("__DEFAULT__")
            .astype(str)
            .values
        )
        cal_factors = np.ones(len(forecasts), dtype=float)
        for i, (h, seg) in enumerate(zip(forecasts["horizon"].values, segs)):
            fmap = artifacts.calibration.get(int(h), {})
            f = fmap.get(seg, fmap.get("__GLOBAL__", 1.0))
            cal_factors[i] = f
        forecasts["calibration_factor"] = cal_factors
        forecasts["predicted"] = np.clip(forecasts["predicted_raw"].values * cal_factors, 0, None)
    else:
        forecasts["calibration_factor"] = 1.0
        forecasts["predicted"] = forecasts["predicted_raw"].clip(lower=0)

    return forecasts


# ---------------------------------------------------------------------------
# Minimal metric helpers (no sklearn)
# ---------------------------------------------------------------------------


def _wmape(a: np.ndarray, f: np.ndarray) -> float:
    a = np.asarray(a, float)
    f = np.asarray(f, float)
    denom = np.sum(np.abs(a))
    return float(np.sum(np.abs(f - a)) / denom) if denom > 1e-9 else float("nan")


def _bias(a: np.ndarray, f: np.ndarray) -> float:
    a = np.asarray(a, float)
    f = np.asarray(f, float)
    denom = np.sum(np.abs(a))
    return float(np.sum(f - a) / denom) if denom > 1e-9 else float("nan")


# ---------------------------------------------------------------------------
# End-to-end convenience entry point
# ---------------------------------------------------------------------------


def run_direct_multihorizon_pipeline(
    train_features_path: str,
    val_features_path: str,
    test_features_path: str,
    origin_week: int,
    cfg: DirectMHConfig,
    segments_csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, DirectMHArtifacts]:
    """Load features, train, predict, and optionally save artifacts.

    Returns (forecasts_df, artifacts).
    """
    logger.info("Loading features ...")
    t0 = time.time()
    # Format-agnostic load: feature engineering now writes parquet by
    # default but the CLI accepts arbitrary paths, so we route per-file
    # based on the suffix.  load_csv (utils/agent_utilities.py) was
    # made parquet-aware previously, so we just delegate.
    from utils.agent_utilities import load_csv as _smart_load
    train = _smart_load(train_features_path, low_memory=False) if train_features_path.lower().endswith('.csv') else pd.read_parquet(train_features_path)
    val   = _smart_load(val_features_path,   low_memory=False) if val_features_path.lower().endswith('.csv')   else pd.read_parquet(val_features_path)
    test  = _smart_load(test_features_path,  low_memory=False) if test_features_path.lower().endswith('.csv')  else pd.read_parquet(test_features_path)
    logger.info(
        "Loaded: train=%s, val=%s, test=%s in %.1fs",
        len(train),
        len(val),
        len(test),
        time.time() - t0,
    )

    key_to_segment: Dict[str, str] = {}
    if segments_csv_path and os.path.exists(segments_csv_path):
        seg = pd.read_csv(segments_csv_path)
        if cfg.segment_col in seg.columns:
            key_to_segment = dict(
                zip(seg[cfg.key_col].astype(str), seg[cfg.segment_col].astype(str))
            )
            logger.info("Loaded %s key->segment mappings", len(key_to_segment))

    artifacts = train_direct_multihorizon(train, val, cfg, key_to_segment=key_to_segment)

    forecasts = predict_direct_multihorizon(
        test,
        artifacts,
        cfg,
        origin_week=origin_week,
        key_to_segment=key_to_segment,
        apply_calibration=cfg.enable_bias_calibration,
        fallback_features=val,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        artifacts.save(os.path.join(output_dir, "direct_mh"))
        fpath = os.path.join(output_dir, "direct_mh_forecasts.csv")
        forecasts.to_csv(fpath, index=False)
        logger.info("Saved forecasts -> %s", fpath)

    return forecasts, artifacts
