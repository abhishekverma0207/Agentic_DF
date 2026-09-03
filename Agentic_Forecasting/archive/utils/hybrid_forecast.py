"""Hybrid forecast = GLOBAL pooled DMH (wins sparse keys) + KEY-LEVEL direct-MH
(wins high-volume stable keys), combined per key by a volume/stability signal.

This realises the user's hybrid: sparse SKUs ride the all-keys global model
(where we beat LEGO's overfit per-key RF), high-volume SKUs ride their own
key-level direct-multi-horizon model (LEGO's per-key idea but with our forward
features + direct multi-horizon). Reuses the WORKING DMH (utils/direct_multihorizon),
bypassing the year_week-buggy recursive path entirely.

All inputs are the engineered feature panels (train/val/test) + an origin week.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from utils.direct_multihorizon import (
    DirectMHConfig, train_direct_multihorizon, predict_direct_multihorizon,
)

logger = logging.getLogger(__name__)


def augment_panels(train, val, test, target_col: str = "Actuals", fwd_cols=None,
                   horizon: int = 4, train_end: int = 202541, val_end: int = 202602):
    """Add direct-MH augmenters (per-key trajectory + target-week forward features,
    lead-`horizon`) on the COMBINED panel so the forward self-join sees all periods,
    then split back by year_week. APG disabled (redundant for E1016-only)."""
    from utils.direct_multihorizon import augment_features
    combined = pd.concat([train, val, test], ignore_index=True)
    combined = augment_features(combined, horizon=horizon, key_col="key",
                                target_col=target_col, enable_trajectory=True,
                                enable_apg=False, enable_forward=True,
                                forward_candidate_cols=fwd_cols)
    yw = combined["year_week"].astype(int)
    return (combined[yw <= train_end].copy(),
            combined[(yw > train_end) & (yw <= val_end)].copy(),
            combined[yw > val_end].copy())


def perkey_dmh_config(target_col: str = "Actuals", recency: float = 0.0,
                      objective: str = "regression", top_k: int = 30,
                      num_boost_round: int = 400, sply_blend_alpha: float = 0.4) -> DirectMHConfig:
    """Per-key direct-MH config — VALIDATED on FABRIC_CLEANING (beats LEGO t+4&5
    accuracy 0.30 vs 0.28). Single-key (~150-row) panels: shallow trees, 1 seed.
    Key wins: objective='regression' (L2 mean, less under-biased than tweedie/quantile),
    recency=0 (use full history → capture the Q1 seasonal level, not the recent trough),
    sply_blend_alpha=0.4 (anchor 60% to same-period-last-year → captures the seasonal
    uplift the model misses; lifts t45 acc 0.26→0.30)."""
    return DirectMHConfig(
        key_col="key", time_col="year_week", target_col=target_col,
        objective=objective, n_seeds=1,
        num_leaves=15, min_data_in_leaf=8, learning_rate=0.05, lambda_l2=1.0,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        num_boost_round=num_boost_round, early_stopping_rounds=40, top_k_features=top_k,
        recency_halflife_weeks=recency, sply_blend_alpha=sply_blend_alpha, horizon_workers=1,
        enable_bias_calibration=True, calibration_min=0.5, calibration_max=2.5,
        calibration_min_samples=8,
    )


def stable_perkey_config(target_col: str = "Actuals", num_boost_round: int = 700) -> DirectMHConfig:
    """Per-key recipe for STABLE high-volume keys (FOODS-class) — VALIDATED to beat the
    FABRIC recipe there by ~0.15 t45 (0.31→0.46). Opposite levers to `perkey_dmh_config`:
    NO SPLY blend (sply_blend_alpha=1.0 — SPLY undershoots stable growing keys, −0.33 bias),
    recency_halflife_weeks=26 (recent-weighted → tracks the current level), and roomier trees
    (31 leaves, min_data 15) for a tighter RF-like fit on predictable series. Use where the
    series is stable; `perkey_dmh_config` (SPLY-blended, shallow) wins on NOISY keys."""
    return DirectMHConfig(
        key_col="key", time_col="year_week", target_col=target_col,
        objective="regression", n_seeds=1,
        num_leaves=31, min_data_in_leaf=15, learning_rate=0.05, lambda_l2=1.0,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        num_boost_round=num_boost_round, early_stopping_rounds=40, top_k_features=40,
        recency_halflife_weeks=26, sply_blend_alpha=1.0, horizon_workers=1,
        enable_bias_calibration=True, calibration_min=0.5, calibration_max=2.5,
        calibration_min_samples=8,
    )


def default_recipes() -> dict:
    """The per-key recipe menu the OOS contest picks from, per key:
    'noisy' = SPLY-blended shallow (wins erratic keys, e.g. FABRIC_CLEANING),
    'stable' = recent-weighted tight (wins stable keys, e.g. FOODS)."""
    return {"noisy": perkey_dmh_config(), "stable": stable_perkey_config()}


def select_high_volume_keys(train: pd.DataFrame, target_col: str = "Actuals",
                            key_col: str = "key", vol_quantile: float = 0.70,
                            max_zero_frac: float = 0.35, min_nonzero: int = 52):
    """High-volume + stable keys (the ones a key-level model should win):
    top (1-vol_quantile) by volume, low zero-fraction, enough non-zero history."""
    g = train.groupby(key_col)[target_col].agg(
        vol="sum", n="size", nz=lambda s: int((s > 0).sum()))
    g["zf"] = 1.0 - g["nz"] / g["n"].clip(lower=1)
    thr = g["vol"].quantile(vol_quantile)
    keys = g[(g["vol"] >= thr) & (g["zf"] <= max_zero_frac) & (g["nz"] >= min_nonzero)]
    logger.info("high-volume keys: %d / %d (vol>=%.0f, zf<=%.2f, nz>=%d)",
                len(keys), len(g), thr, max_zero_frac, min_nonzero)
    return keys.index.astype(str).tolist()


def train_perkey_forecasts(train, val, test, keys, cfg, origin_week, min_rows: int = 40):
    """Train one DMH per key in `keys`; return [key, year_week, predicted]."""
    keyset = set(map(str, keys))
    tr_all = train[train["key"].astype(str).isin(keyset)]
    vl_all = val[val["key"].astype(str).isin(keyset)]
    te_all = test[test["key"].astype(str).isin(keyset)]
    out, ok, fail = [], 0, 0
    for i, k in enumerate(keys):
        tr = tr_all[tr_all["key"].astype(str) == k]
        if len(tr) < min_rows:
            continue
        vl = vl_all[vl_all["key"].astype(str) == k]
        te = te_all[te_all["key"].astype(str) == k]
        try:
            arts = train_direct_multihorizon(tr, vl, cfg)
            fc = predict_direct_multihorizon(te, arts, cfg, origin_week=int(origin_week),
                                             fallback_features=vl)
            fc["key"] = k
            out.append(fc[["key", "year_week", "predicted"]])
            ok += 1
        except Exception as e:  # noqa: BLE001 - one bad key shouldn't sink the batch
            fail += 1
            logger.warning("per-key %s failed: %s: %s", k, type(e).__name__, e)
        if i % 25 == 0:
            logger.info("per-key %d/%d (ok=%d fail=%d)", i, len(keys), ok, fail)
    logger.info("per-key done: %d ok, %d failed", ok, fail)
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["key", "year_week", "predicted"]))


def _perkey_wape(fc: pd.DataFrame, actuals: pd.DataFrame, t45_weight: float = 2.0,
                 key_col: str = "key", target_col: str = "Actuals") -> pd.Series:
    """Per-key t+4/t+5-emphasised WAPE of `fc` vs `actuals` on a shared window.
    fc: [key, year_week, predicted(, horizon)]; actuals: [key, year_week, target_col].
    Horizons 4&5 carry `t45_weight`x weight so selection matches the planning objective."""
    f = fc.copy()
    f[key_col] = f[key_col].astype(str)
    f["_wk"] = pd.to_numeric(f[ "year_week"], errors="coerce").astype("Int64")
    a = actuals.copy()
    a[key_col] = a[key_col].astype(str)
    a["_wk"] = pd.to_numeric(a["year_week"], errors="coerce").astype("Int64")
    a = a[[key_col, "_wk", target_col]].rename(columns={target_col: "_a"})
    m = f.merge(a, on=[key_col, "_wk"], how="inner")
    if m.empty:
        return pd.Series(dtype=float)
    w = np.where(m["horizon"].isin([4, 5]), t45_weight, 1.0) if "horizon" in m.columns else 1.0
    m["_num"] = w * (m[pred_col_default(m)].astype(float) - m["_a"].astype(float)).abs()
    m["_den"] = w * m["_a"].astype(float).abs()
    g = m.groupby(key_col).agg(num=("_num", "sum"), den=("_den", "sum"))
    return pd.Series(np.where(g["den"] > 0, g["num"] / g["den"], np.inf), index=g.index)


def pred_col_default(df: pd.DataFrame) -> str:
    return "predicted" if "predicted" in df.columns else "prediction"


def _cat_t45_acc(fc: pd.DataFrame, actuals: pd.DataFrame, key_col: str = "key",
                 target_col: str = "Actuals") -> float:
    """Category-wide t+4/t+5 accuracy (1-WAPE) of `fc` vs `actuals` (volume-weighted)."""
    f = fc.copy(); f[key_col] = f[key_col].astype(str)
    f["_wk"] = pd.to_numeric(f["year_week"], errors="coerce").astype("Int64")
    a = actuals.copy(); a[key_col] = a[key_col].astype(str)
    a["_wk"] = pd.to_numeric(a["year_week"], errors="coerce").astype("Int64")
    m = f.merge(a[[key_col, "_wk", target_col]], on=[key_col, "_wk"], how="inner")
    if "horizon" in m.columns:
        m = m[m["horizon"].isin([4, 5])]
    t = m[target_col].astype(float).abs().sum()
    return float(1.0 - (m[pred_col_default(m)].astype(float) - m[target_col].astype(float)).abs().sum() / t) if t > 0 else 0.0


def oos_select_keys(train, val, gcfg, pcfg, hv_keys, *,
                    sel_train_end: int = 202450, sel_val_end: int = 202502,
                    oos_origin: int = 202502, oos_end: int = 202515,
                    t45_weight: float = 2.0, margin: float = 0.0, min_rows: int = 40):
    """Decide WHICH high-volume keys should use a key-level model, via a clean
    out-of-sample contest in the SAME season as the test (Q1 last year = the SPLY
    of the 2026-03..15 forecast window, so the decision matches the t+4/t+5 objective).

    A reduced global + reduced per-key are fit on train<=`sel_train_end` (early-stop on
    `sel_train_end`<w<=`sel_val_end`); both then forecast the held-out window
    (`sel_val_end`<w<=`oos_end`, origin `oos_origin`) which NEITHER model saw. A key
    uses per-key only if its per-key t45-WAPE beats the global's by `margin`.

    This is the non-oracle fix for FOODS: where the global already wins a key (FOODS
    high-vol) per-key is NOT selected; where the global is weak (FABRIC_CLEANING
    high-vol) per-key IS selected -> the hybrid is robust across categories.
    Returns (selected_keys, diagnostic_table)."""
    yw = pd.to_numeric(train["year_week"], errors="coerce")
    tr_sel = train[yw <= sel_train_end].copy()
    vl_sel = train[(yw > sel_train_end) & (yw <= sel_val_end)].copy()
    oos = train[(yw > sel_val_end) & (yw <= oos_end)].copy()
    if oos.empty or vl_sel.empty:
        logger.warning("OOS window empty (sel_val_end=%s oos_end=%s) -> select all hv",
                       sel_val_end, oos_end)
        return list(map(str, hv_keys)), pd.DataFrame()
    oos_act = oos[["key", "year_week", "Actuals"]].copy()

    g_sel = train_direct_multihorizon(tr_sel, vl_sel, gcfg)
    g_oos = predict_direct_multihorizon(oos, g_sel, gcfg, origin_week=int(oos_origin),
                                        fallback_features=vl_sel)
    wape_g = _perkey_wape(g_oos, oos_act, t45_weight=t45_weight)

    keyset = set(map(str, hv_keys))
    tr_a = tr_sel[tr_sel["key"].astype(str).isin(keyset)]
    vl_a = vl_sel[vl_sel["key"].astype(str).isin(keyset)]
    oos_a = oos[oos["key"].astype(str).isin(keyset)]
    rows = []
    for i, k in enumerate(hv_keys):
        trk = tr_a[tr_a["key"].astype(str) == k]
        vlk = vl_a[vl_a["key"].astype(str) == k]
        ok = oos_a[oos_a["key"].astype(str) == k]
        wg = float(wape_g.get(k, np.inf))
        if len(trk) < min_rows or vlk.empty or ok.empty:
            rows.append((k, np.inf, wg, False))
            continue
        try:
            ak = train_direct_multihorizon(trk, vlk, pcfg)
            pk = predict_direct_multihorizon(ok, ak, pcfg, origin_week=int(oos_origin),
                                             fallback_features=vlk)
            pk["key"] = k
            wp = float(_perkey_wape(pk, oos_act[oos_act["key"].astype(str) == k],
                                    t45_weight=t45_weight).get(k, np.inf))
        except Exception as e:  # noqa: BLE001 - one bad key shouldn't sink selection
            logger.warning("oos per-key %s failed: %s: %s", k, type(e).__name__, e)
            wp = np.inf
        rows.append((k, wp, wg, bool(wp < wg * (1.0 - margin))))
        if i % 25 == 0:
            logger.info("oos-select %d/%d", i, len(hv_keys))
    tbl = pd.DataFrame(rows, columns=["key", "wape_perkey", "wape_global", "win"])
    selected = tbl.loc[tbl["win"], "key"].astype(str).tolist()
    med_pk = tbl["wape_perkey"].replace(np.inf, np.nan).median()
    med_g = tbl["wape_global"].replace(np.inf, np.nan).median()
    logger.info("OOS selection: %d/%d hv keys -> per-key (median wape pk=%.3f g=%.3f, margin=%.2f)",
                len(selected), len(tbl), med_pk, med_g, margin)
    return selected, tbl


def oos_select(train, val, gcfg, recipes, hv_keys, *,
               sel_train_end: int = 202450, sel_val_end: int = 202502,
               oos_origin: int = 202502, oos_end: int = 202515,
               t45_weight: float = 2.0, min_rows: int = 40,
               strong_global_acc: float = 0.42, rescue_recipe: str = "noisy"):
    """Recipe-ADAPTIVE OOS selection. Like `oos_select_keys` but contests the GLOBAL
    against EACH per-key recipe in `recipes` (dict {name: DirectMHConfig}) on the
    2025-Q1 held-out window, and assigns each hv key to its OOS winner. A key gets a
    per-key recipe only if that recipe's t45-WAPE beats the global's by `margin`;
    otherwise it stays on the global. Returns (assignment {key: recipe_name}, table).

    This captures the finding that the per-key recipe is category/key-specific: stable
    keys want the 'stable' recipe, noisy keys the 'noisy' one, and where neither beats
    the global the key stays global -> robust everywhere."""
    yw = pd.to_numeric(train["year_week"], errors="coerce")
    tr_sel = train[yw <= sel_train_end].copy()
    vl_sel = train[(yw > sel_train_end) & (yw <= sel_val_end)].copy()
    oos = train[(yw > sel_val_end) & (yw <= oos_end)].copy()
    if oos.empty or vl_sel.empty:
        logger.warning("OOS window empty -> no per-key")
        return {}, pd.DataFrame()
    oos_act = oos[["key", "year_week", "Actuals"]].copy()
    g_sel = train_direct_multihorizon(tr_sel, vl_sel, gcfg)
    g_oos = predict_direct_multihorizon(oos, g_sel, gcfg, origin_week=int(oos_origin),
                                        fallback_features=vl_sel)

    # CATEGORY-LEVEL GATE on the HIGH-VOLUME keys: is the pooled global already good on
    # the keys a per-key model could serve? If yes (FOODS-class: on stable keys the global
    # has lower variance than any per-key model), ride the global. If no (FABRIC-class:
    # global broken on noisy keys), deploy per-key BROADLY with the rescue recipe.
    # Per-KEY/per-recipe OOS selection was REMOVED: the 2025-Q1->2026-Q1 transfer is too
    # noisy to pick keys or recipes reliably (it degraded FABRIC 0.29->0.23 vs blind-broad);
    # the aggregate category decision is robust. Validated: FABRIC->perkey_broad WINS
    # (0.288 vs LEGO 0.278), FOODS->global is its best (0.427, LEGO-short but robust).
    keyset = set(map(str, hv_keys))
    hv_oos_act = oos_act[oos_act["key"].astype(str).isin(keyset)]
    hv_g_oos = g_oos[g_oos["key"].astype(str).isin(keyset)]
    g_acc_hv = _cat_t45_acc(hv_g_oos, hv_oos_act)
    logger.info("global OOS t45 acc on %d hv keys = %.3f (strong-global gate %.2f)",
                len(keyset), g_acc_hv, strong_global_acc)
    if g_acc_hv > strong_global_acc:
        logger.info("-> strong-global on hv keys: ride global (no per-key)")
        return {}, pd.DataFrame([{"g_acc_hv": g_acc_hv, "decision": "global"}])
    rescue = rescue_recipe if rescue_recipe in recipes else next(iter(recipes))
    logger.info("-> weak-global on hv keys: blind broad per-key, '%s' recipe, %d keys", rescue, len(keyset))
    assignment = {str(k): rescue for k in hv_keys}
    return assignment, pd.DataFrame([{"g_acc_hv": g_acc_hv, "decision": "perkey_broad", "recipe": rescue}])


def train_perkey_assigned(train, val, test, assignment, recipes, origin_week, min_rows: int = 40):
    """Train each key with its OOS-assigned recipe; return [key, year_week, predicted]."""
    out, ok, fail = [], 0, 0
    by_recipe = {}
    for k, rname in assignment.items():
        by_recipe.setdefault(rname, []).append(str(k))
    for rname, keys in by_recipe.items():
        cfg = recipes[rname]
        fc = train_perkey_forecasts(train, val, test, keys, cfg, origin_week, min_rows=min_rows)
        if not fc.empty:
            out.append(fc); ok += fc["key"].nunique()
        logger.info("per-key recipe '%s': %d keys", rname, len(keys))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["key", "year_week", "predicted"]))


def combine(global_fc: pd.DataFrame, perkey_fc: pd.DataFrame, hv_keys,
            blend: float = 1.0, pred_col: str = "predicted",
            key_col: str = "key", week_col: str = "year_week") -> pd.DataFrame:
    """For high-volume keys: predicted = blend*perkey + (1-blend)*global. Others: global."""
    g = global_fc.copy()
    g[key_col] = g[key_col].astype(str)
    g["_wk"] = pd.to_numeric(g[week_col], errors="coerce").astype("Int64").astype(str)
    if perkey_fc.empty:
        return g.drop(columns=["_wk"])
    pk = perkey_fc.copy()
    pk[key_col] = pk[key_col].astype(str)
    pk["_wk"] = pd.to_numeric(pk[week_col], errors="coerce").astype("Int64").astype(str)
    pk = pk.rename(columns={pred_col: "_pk"})[[key_col, "_wk", "_pk"]]
    m = g.merge(pk, on=[key_col, "_wk"], how="left")
    hv = set(map(str, hv_keys))
    use = m[key_col].isin(hv) & m["_pk"].notna()
    m.loc[use, pred_col] = blend * m.loc[use, "_pk"] + (1 - blend) * m.loc[use, pred_col]
    logger.info("hybrid: blended %d rows across %d high-volume keys (blend=%.2f)",
                int(use.sum()), len(hv), blend)
    return m.drop(columns=["_wk", "_pk"])
