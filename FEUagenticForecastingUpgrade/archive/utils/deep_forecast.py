"""Deep multi-horizon forecaster (v2) — production module.

A GLOBAL Temporal Fusion Transformer (neuralforecast) forecasts all H horizons
JOINTLY across every series, with known-future inputs (calendar/holiday/promo) and
static category covariates. This is the balanced-across-the-horizon model that beat
the per-horizon GBMs on the high-volume TH categories (e.g. FOODS) at both t+4&5 and
13-week.

Two public entry points:
  * ``train_global_deep(df, ...)`` -> forecast DataFrame [key, period, predicted].
  * ``apply_deep_overlay(full_df, category_artifact_paths, cfg, ...)`` -> the DIQ-runner
    hook: trains once across the routed categories and OVERWRITES the ``predicted``
    column of each routed category's ``model_artifacts/inference_forecast.csv`` (schema
    preserved so the downstream stack + MOQ are unchanged).

All neuralforecast/torch imports are lazy so importing this module (and the diq_runner)
never requires the deep stack unless the deep overlay is actually switched on.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# known-future covariate name patterns (promo/holiday/calendar — NOT sales-derived,
# so safe as future regressors). Same set validated in the local v2 build.
_FUTR_PATTERNS = ["songkran", "bucha", "festive", "holiday", "promo", "discount",
                  "season", "priceoff", "off_invoice", "stringency", "summer",
                  "monsoon", "winter"]
_MAX_FUTR = 14


def _to_ds(period: pd.Series) -> pd.Series:
    """YYYYWW (int-like) -> ISO-week Monday timestamp."""
    s = pd.to_numeric(period, errors="coerce")
    return pd.to_datetime(s.astype("Int64").astype(str) + "1", format="%G%V%u")


def _to_period(ds: pd.Series) -> pd.Series:
    iso = ds.dt.isocalendar()
    return (iso["year"] * 100 + iso["week"]).astype(int)


def train_global_deep(
    df: pd.DataFrame,
    *,
    key_col: str = "key",
    period_col: str = "year_week",
    target_col: str = "Actuals",
    category_col: Optional[str] = "Category",
    origin_period: int,
    horizon: int = 13,
    input_size: int = 52,
    model_name: str = "TFT",
    max_steps: int = 2000,
    hidden_size: int = 96,
    min_history: int = 40,
    accelerator: str = "auto",
    loss_name: str = "MAE",
    extra_static_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Train one global deep model on ``df`` (history <= origin_period) and forecast
    ``horizon`` periods. Returns [key_col, period_col, 'predicted']. Lazy-imports
    neuralforecast/torch so the module stays import-safe without the deep stack."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
    import warnings; warnings.filterwarnings("ignore")
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, TFT, TiDE
    from neuralforecast.losses.pytorch import MAE, RMSE, MSE, HuberLoss
    # mean-targeting losses (RMSE/MSE) lift the forecast off the median (MAE) toward the
    # mean -> reduces the systematic under-bias on right-skewed demand (matches LEGO's RF/MSE).
    _LOSSES = {"MAE": MAE, "RMSE": RMSE, "MSE": MSE, "Huber": HuberLoss}
    if loss_name.startswith("Huber") and loss_name != "Huber":
        loss_obj = HuberLoss(delta=float(loss_name[5:]))   # e.g. "Huber5" -> HuberLoss(delta=5)
    else:
        loss_obj = _LOSSES.get(loss_name, MAE)()

    d = df[[c for c in df.columns]].copy()
    d["__p"] = pd.to_numeric(d[period_col], errors="coerce")
    d = d[d["__p"].notna()].copy()
    d["unique_id"] = d[key_col].astype(str)
    d["ds"] = _to_ds(d["__p"])
    # keep history + exactly `horizon` future weeks. Date arithmetic is year-boundary safe;
    # the integer `origin_period + horizon` breaks across the year (e.g. 202541+13 = 202554,
    # a non-existent week 54, dropping the real future weeks 202601/202602).
    _cut = pd.to_datetime(str(int(origin_period)) + "1", format="%G%V%u") + pd.Timedelta(weeks=horizon)
    d = d[d["ds"] <= _cut].copy()
    d["y"] = pd.to_numeric(d[target_col], errors="coerce").fillna(0).clip(lower=0)
    # calendar (always-known) + curated promo/holiday covariates
    woy = d["ds"].dt.isocalendar().week.astype(int)
    d["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    d["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    d["is_q1"] = (woy <= 15).astype(float)
    fut_mask = d["__p"] > origin_period
    cov = [c for c in d.columns if any(p in c.lower() for p in _FUTR_PATTERNS)
           and pd.api.types.is_numeric_dtype(d[c]) and d[c].notna().any()]
    cov = [c for c in cov if d.loc[fut_mask, c].notna().any() and float(d[c].std() or 0) > 1e-6][:_MAX_FUTR]
    futr = ["woy_sin", "woy_cos", "is_q1"] + cov
    for c in futr:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    # static: category one-hot + log mean volume
    if category_col and category_col in d.columns:
        cat_oh = pd.get_dummies(d.groupby("unique_id")[category_col].first()).astype(float)
    else:
        cat_oh = pd.DataFrame(index=d["unique_id"].unique())
    g = d.groupby(["unique_id", "ds"], as_index=False).agg(
        {"y": "sum", "__p": "first", **{c: "first" for c in futr}})
    vol = np.log1p(g[g["__p"] <= origin_period].groupby("unique_id")["y"].mean())
    static = cat_oh.copy(); static["logvol"] = vol.reindex(static.index).fillna(0)
    # OPTIONAL extra per-key static covariates (forecastability features: ADI, CV2, spike,
    # recent-level, etc.) so the global model can DIFFERENTIATE series the way a per-key RF does.
    if extra_static_df is not None and len(extra_static_df):
        ex = extra_static_df.copy(); ex.index = ex.index.astype(str)
        ex = ex.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        for c in ex.columns:
            static[f"sx_{c}"] = ex[c].reindex(static.index).astype(float)
        static[[f"sx_{c}" for c in ex.columns]] = static[[f"sx_{c}" for c in ex.columns]].fillna(
            static[[f"sx_{c}" for c in ex.columns]].median())
    static.index.name = "unique_id"; static = static.reset_index()
    stat_cols = [c for c in static.columns if c != "unique_id"]

    train = g[g["__p"] <= origin_period][["unique_id", "ds", "y"] + futr]
    futr_df = g[g["__p"] > origin_period][["unique_id", "ds"] + futr]
    good = futr_df.groupby("unique_id").size(); good = good[good == horizon].index
    hl = train.groupby("unique_id").size(); good = good.intersection(hl[hl >= min_history].index)
    train = train[train["unique_id"].isin(good)]
    futr_df = futr_df[futr_df["unique_id"].isin(good)]
    static = static[static["unique_id"].isin(good)]
    logger.info("[deep] %s: %d series, %d futr covs, %d static (origin=%s, h=%d)",
                model_name, train["unique_id"].nunique(), len(futr), len(stat_cols),
                origin_period, horizon)
    if train["unique_id"].nunique() == 0:
        return pd.DataFrame(columns=[key_col, period_col, "predicted"])

    common = dict(h=horizon, input_size=input_size, futr_exog_list=futr,
                  stat_exog_list=stat_cols, scaler_type="robust", loss=loss_obj,
                  max_steps=max_steps, val_check_steps=50, early_stop_patience_steps=4,
                  enable_progress_bar=False, logger=False, start_padding_enabled=True,
                  batch_size=32)
    if accelerator != "auto":
        common.update(accelerator=accelerator, devices=1)
    if model_name == "NHITS":
        model = NHITS(**common, n_pool_kernel_size=[2, 2, 1], n_freq_downsample=[4, 2, 1])
    elif model_name == "TiDE":
        model = TiDE(**common)
    else:
        model = TFT(**common, hidden_size=hidden_size, n_head=4)
    nf = NeuralForecast(models=[model], freq="W-MON")
    nf.fit(df=train, static_df=static, val_size=horizon)
    fc = nf.predict(futr_df=futr_df)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    out = pd.DataFrame({
        key_col: fc["unique_id"].astype(str),
        period_col: _to_period(fc["ds"]).astype(int),
        "predicted": fc[model_name].clip(lower=0).astype(float),
    })
    return out


def apply_deep_overlay(
    full_df: pd.DataFrame,
    category_artifact_paths: Dict[str, Path],
    cfg,
    *,
    snapshot_week,
    route_categories: Sequence[str],
    model_name: str = "TFT",
    max_steps: int = 2000,
    horizon: Optional[int] = None,
    category_col: str = "category_name",
) -> List[str]:
    """DIQ-runner hook. Trains one global deep model across ALL series in ``full_df``,
    then for each category in ``route_categories`` overwrites the ``predicted`` column
    of that category's ``model_artifacts/inference_forecast.csv`` with the deep forecast
    (matched on key x period). Schema is preserved -> downstream stack + MOQ unchanged.
    Returns the list of categories actually overwritten. Best-effort: any failure leaves
    the GBM forecast in place (so the run never regresses below the GBM baseline)."""
    key_col = (cfg.prediction_key_cols or ["key"])[0]
    period_col = cfg.timestamp_col
    target_col = cfg.target_col
    h = int(horizon or cfg.forecast_horizon)
    origin = int(str(snapshot_week)) - 1  # last history period before the snapshot
    route = {str(c).strip().upper() for c in route_categories}
    cat_col = category_col if category_col in full_df.columns else (
        "Category" if "Category" in full_df.columns else None)

    try:
        fc = train_global_deep(
            full_df, key_col=key_col, period_col=period_col, target_col=target_col,
            category_col=cat_col, origin_period=origin, horizon=h,
            model_name=model_name, max_steps=max_steps)
    except Exception as exc:  # noqa: BLE001 - never let the deep step sink the run
        logger.error("[deep] overlay training failed (%s) — keeping GBM forecasts", exc, exc_info=True)
        return []
    if fc.empty:
        logger.warning("[deep] overlay produced no forecasts — keeping GBM forecasts")
        return []
    fc["__k"] = fc[key_col].astype(str)
    fc["__p"] = pd.to_numeric(fc[period_col], errors="coerce").astype("Int64")
    deep_by_key = fc.set_index(["__k", "__p"])["predicted"]

    overwritten = []
    for cat, art_dir in category_artifact_paths.items():
        if cat.strip().upper() not in route:
            continue
        csv_path = Path(art_dir) / "model_artifacts" / "inference_forecast.csv"
        if not csv_path.exists():
            logger.warning("[deep] %s: inference_forecast.csv missing — skip overlay", cat)
            continue
        try:
            gbm = pd.read_csv(csv_path)
            k = gbm[key_col].astype(str)
            p = pd.to_numeric(gbm[period_col], errors="coerce").astype("Int64")
            new = deep_by_key.reindex(list(zip(k, p))).values
            mask = ~pd.isna(new)
            if mask.sum() == 0:
                logger.warning("[deep] %s: no key x period overlap — keeping GBM", cat)
                continue
            gbm.loc[mask, "predicted"] = np.asarray(new, dtype="float64")[mask]
            gbm.to_csv(csv_path, index=False)
            overwritten.append(cat)
            logger.info("[deep] %s: overlaid %d/%d rows with %s forecast",
                        cat, int(mask.sum()), len(gbm), model_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("[deep] %s: overlay write failed (%s) — keeping GBM", cat, exc)
    logger.info("[deep] overlay complete: %d/%d routed categories overwritten",
                len(overwritten), len(route))
    return overwritten


__all__ = ["train_global_deep", "apply_deep_overlay"]
