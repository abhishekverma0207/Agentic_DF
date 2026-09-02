"""LEGO scoreboard — score our Thailand forecast against the LEGO benchmark.

Eval grain (verified): LEGO `key` = <CS_BARCODE>_E1016; E1016 is ONE account, and
LEGO `target` == TH `Actuals` for account E1016 (exact). So we:
  1. take OUR forecast, keep E1016 keys, strip "_E1016" -> CS_BARCODE,
  2. aggregate predictions to CS_BARCODE x week,
  3. join to the cached eval panel (actual=th_e1016, LEGO=prediction_rf),
  4. report per-category x per-horizon 1-WAPE & bias for Ours vs LEGO vs level8,
     with PASS/FAIL (beat = higher 1-WAPE AND smaller |bias|), t+4/t+5 emphasised.

Reuses cached artifacts in notebooks/rca_outputs/ (no 4.5GB rescan):
  - lego_eval_joined.parquet : cs, Shipment_Week, prediction_rf, th_e1016, Category, t
  - th_cs_week_actuals.parquet: cs, Shipment_Week, th_e1016 (for the level8 baseline)

CLI:  .venv/bin/python -m utils.lego_eval <forecast.parquet|csv> [--pred-col predicted]
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

RCA = "notebooks/rca_outputs"
CUTOFF = "2026-02"
EVAL_WEEKS = [f"2026-{w:02d}" for w in range(3, 16)]   # t+1..t+13
HORIZON_OF_WEEK = {w: i + 1 for i, w in enumerate(EVAL_WEEKS)}
PRIORITY = [4, 5]                                       # t+4, t+5 emphasis


# ----------------------------------------------------------------- metrics
def _acc_bias(a: np.ndarray, f: np.ndarray):
    """Return (1-WAPE, bias, sum_actual). WAPE/bias are pooled ratios of sums."""
    a = np.nan_to_num(a, nan=0.0)
    f = np.nan_to_num(f, nan=0.0)
    sa = a.sum()
    if sa <= 0:
        return np.nan, np.nan, sa
    return 1.0 - np.abs(f - a).sum() / sa, (f - a).sum() / sa, sa


def _norm_week(s: pd.Series) -> pd.Series:
    """Zero-pad dashed YYYY-WW so string compares/joins line up (2026-3 -> 2026-03)."""
    def n(x):
        x = str(x).strip()
        if "-" in x:
            y, w = x.split("-", 1)
            if y.isdigit() and w.isdigit():
                return f"{y}-{w.zfill(2)}"
        if x.isdigit() and len(x) == 6:          # YYYYWW -> YYYY-WW
            return f"{x[:4]}-{x[4:]}"
        return x
    return s.map(n)


# ----------------------------------------------------------------- panels
def load_eval_base() -> pd.DataFrame:
    base = pd.read_parquet(f"{RCA}/lego_eval_joined.parquet")
    base = base[["cs", "Shipment_Week", "prediction_rf", "th_e1016", "Category", "t"]].copy()
    base["Shipment_Week"] = _norm_week(base["Shipment_Week"])
    base["actual"] = pd.to_numeric(base["th_e1016"], errors="coerce").fillna(0.0)
    base["lego"] = pd.to_numeric(base["prediction_rf"], errors="coerce").fillna(0.0)
    # level8 baseline = mean E1016 of last 8 weeks <= cutoff, per cs
    th = pd.read_parquet(f"{RCA}/th_cs_week_actuals.parquet")
    th["Shipment_Week"] = _norm_week(th["Shipment_Week"])
    recent = th[(th["Shipment_Week"] > "2025-46") & (th["Shipment_Week"] <= CUTOFF)]
    lvl8 = recent.groupby("cs")["th_e1016"].mean()
    base["level8"] = base["cs"].map(lvl8).fillna(0.0)
    return base


def attach_forecast(base: pd.DataFrame, fc: pd.DataFrame,
                    pred_col: str = "predicted", key_col: str | None = None,
                    week_col: str | None = None) -> pd.DataFrame:
    """Aggregate OUR forecast to cs x week and merge onto the eval base."""
    fc = fc.copy()
    if key_col is None:
        key_col = "key" if "key" in fc.columns else "Model_Hierarchy"
    if week_col is None:
        for c in ("Shipment_Week", "year_week"):
            if c in fc.columns:
                week_col = c
                break
    if week_col is None or key_col not in fc.columns or pred_col not in fc.columns:
        raise ValueError(f"forecast must have {key_col}, a week col, and {pred_col}; "
                         f"got {list(fc.columns)[:15]}")
    fc[key_col] = fc[key_col].astype(str)
    fc = fc[fc[key_col].str.endswith("_E1016")]
    fc["cs"] = fc[key_col].str[:-len("_E1016")]
    fc["Shipment_Week"] = _norm_week(fc[week_col])
    fc = fc[fc["Shipment_Week"].isin(EVAL_WEEKS)]
    fc["pred"] = pd.to_numeric(fc[pred_col], errors="coerce").fillna(0.0)
    ours = fc.groupby(["cs", "Shipment_Week"], as_index=False)["pred"].sum()
    out = base.merge(ours, on=["cs", "Shipment_Week"], how="left")
    out["ours"] = out["pred"].fillna(0.0)
    cov = out["pred"].notna().mean()
    print(f"[lego_eval] our forecast covers {cov:.1%} of the {len(base)} eval cells "
          f"({out['cs'].nunique()} cs)", flush=True)
    return out


# ----------------------------------------------------------------- scoreboard
def _slice(df, group):
    if group == "all13":
        return df
    if group == "t45":
        return df[df["t"].isin(PRIORITY)]
    return df[df["t"] == int(group[1:])]   # 't4' -> t==4


def scoreboard(panel: pd.DataFrame, models=("ours", "lego", "level8")) -> pd.DataFrame:
    """Per-category x horizon-group 1-WAPE & bias for each model + Ours-vs-LEGO verdict."""
    groups = ["t4", "t5", "t45", "all13"]
    cats = (panel.groupby("Category")["actual"].sum().sort_values(ascending=False).index.tolist())
    rows = []
    for cat in cats + ["__ALL__"]:
        sub = panel if cat == "__ALL__" else panel[panel["Category"] == cat]
        for g in groups:
            s = _slice(sub, g)
            rec = {"Category": cat, "group": g, "sum_actual": s["actual"].sum()}
            for m in models:
                acc, bias, _ = _acc_bias(s["actual"].values, s[m].values)
                rec[f"acc_{m}"] = acc
                rec[f"bias_{m}"] = bias
            # verdict vs LEGO
            rec["acc_win"] = rec["acc_ours"] > rec["acc_lego"]
            rec["bias_win"] = abs(rec["bias_ours"]) < abs(rec["bias_lego"])
            rec["WIN"] = bool(rec["acc_win"] and rec["bias_win"])
            rows.append(rec)
    return pd.DataFrame(rows)


def print_scoreboard(sb: pd.DataFrame, label: str = "Ours"):
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.float_format", lambda x: f"{x:7.3f}")
    print("\n" + "=" * 100)
    print(f"LEGO SCOREBOARD — {label} vs LEGO (acc = 1-WAPE; WIN = higher acc AND smaller |bias|)")
    print("=" * 100)
    for cat in sb["Category"].unique():
        c = sb[sb["Category"] == cat]
        print(f"\n### {cat}   (eval sales {c['sum_actual'].iloc[-1]:,.0f})")
        show = c[["group", "acc_ours", "acc_lego", "acc_level8",
                  "bias_ours", "bias_lego", "WIN"]].copy()
        show.columns = ["grp", "acc_OURS", "acc_LEGO", "acc_lvl8",
                        "bias_OURS", "bias_LEGO", "WIN"]
        print(show.to_string(index=False))
    # headline: priority t45 win-rate across real categories
    real = sb[(sb["Category"] != "__ALL__") & (sb["group"] == "t45")]
    allc = sb[(sb["Category"] == "__ALL__")]
    print("\n" + "-" * 100)
    print(f"HEADLINE  t+4&t+5 categories WON: {int(real['WIN'].sum())}/{len(real)}  "
          f"| 13wk categories WON: {int(sb[(sb.Category!='__ALL__')&(sb.group=='all13')]['WIN'].sum())}"
          f"/{sb[(sb.Category!='__ALL__')&(sb.group=='all13')].shape[0]}")
    for _, r in allc.iterrows():
        print(f"  OVERALL {r['group']:6s}: acc OURS={r['acc_ours']:.3f} LEGO={r['acc_lego']:.3f} "
              f"lvl8={r['acc_level8']:.3f} | bias OURS={r['bias_ours']:+.3f} LEGO={r['bias_lego']:+.3f} "
              f"| {'WIN' if r['WIN'] else 'lose'}")
    print("-" * 100, flush=True)


def evaluate(forecast, pred_col="predicted", key_col=None, week_col=None, label="Ours"):
    """Top-level: load base, attach forecast, print + return the scoreboard."""
    if isinstance(forecast, str):
        forecast = pd.read_parquet(forecast) if forecast.endswith(".parquet") else pd.read_csv(forecast)
    base = load_eval_base()
    panel = attach_forecast(base, forecast, pred_col=pred_col, key_col=key_col, week_col=week_col)
    sb = scoreboard(panel)
    print_scoreboard(sb, label=label)
    return sb, panel


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("forecast")
    ap.add_argument("--pred-col", default="predicted")
    ap.add_argument("--key-col", default=None)
    ap.add_argument("--week-col", default=None)
    ap.add_argument("--label", default="Ours")
    a = ap.parse_args()
    evaluate(a.forecast, pred_col=a.pred_col, key_col=a.key_col,
             week_col=a.week_col, label=a.label)
