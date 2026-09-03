#!/usr/bin/env python
"""#1 — can we keep the un-guardrailed blend's accuracy AND fix its under-forecast?

The un-guardrailed blends (raw / lasso) beat LEGO by ~+6.5pp OOS but under-forecast ~4%.
Here we multiplicatively CALIBRATE the blend by a scale estimated ON TRAIN ONLY
(scale = Σactual / Σblend), then score OOS on 202616. Global vs per-category scale.

If per-category calibration keeps the accuracy and zeroes the bias, more categories pass
the acc+bias win rule → genuinely better than both prod (guardrail) and raw.

Run:  .venv/bin/python scripts/de_bias_correct.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.de_stacker_compare as C       # reuse data loader + fitters
import scripts.de_segment_research as R       # reuse metrics + apply_blend

WEEK = R.WEEK
A_LASSO = 0.2     # from the out-of-fold selection in de_stacker_compare


def scale_by(train_df, btr, by):
    """Multiplicative calibration estimated on TRAIN: scale = Σactual / Σblend (clipped)."""
    t = train_df.assign(_b=btr)
    if by == "global":
        return float(np.clip(t["actual"].sum() / max(t["_b"].sum(), 1e-9), 0.5, 2.0))
    g = t.groupby(by).agg(a=("actual", "sum"), b=("_b", "sum"))
    return (g["a"] / g["b"].clip(lower=1e-9)).clip(0.5, 2.0).to_dict()


def apply_scale(df, b, by, sc):
    if by == "global":
        return b * sc
    return b * df[by].map(sc).fillna(1.0).to_numpy(float)


def _cat_bias(s, col):
    g = s.groupby(["cs_gtin", WEEK]).agg(p=(col, "sum"), a=("actual", "sum")).reset_index()
    return R._bias(g.p, g.a)


def wins(e):
    n = 0
    for cat in sorted(e["category"].unique()):
        s = e[e["category"] == cat]
        aL, _ = R._score(s, "rf"); aB, bB = R._score(s, "blend")
        if (aB >= aL - 1e-3) and (min(bB, 0) >= min(_cat_bias(s, "rf"), 0) - 0.02):
            n += 1
    return n


def run_one(train_df, ev, fitted, label, rows, detail=None):
    btr = R.apply_blend(train_df, fitted)
    bev = R.apply_blend(ev, fitted)
    for by in ["none", "global", "category"]:
        bc = bev if by == "none" else apply_scale(ev, bev, by, scale_by(train_df, btr, by))
        e = ev.copy(); e["blend"] = bc
        aL, _ = R._score(e, "rf"); aB, bB = R._score(e, "blend")
        rows.append(dict(label=f"{label}  +cal:{by}", acc=aB, dacc=aB - aL, bias=bB,
                         wins=wins(e), ncats=e["category"].nunique()))
        if detail is not None and label.startswith(detail[0]) and by == detail[1]:
            detail[2] = e.copy()


def cal_then_guardrail(train_df, ev, fitted, margin):
    """Calibrate (global, train-estimated), then per-category guardrail decided ON TRAIN:
    keep the calibrated blend for a category only if it beats LEGO on acc+bias there; else LEGO.
    Apply the train decision OOS. The synthesis of #1 (bias-fix) + a guardrail on the FIXED blend."""
    btr = R.apply_blend(train_df, fitted)
    sc = scale_by(train_df, btr, "global")
    tr = train_df.copy(); tr["blend"] = btr * sc
    keep = {}
    for cat in sorted(tr["category"].unique()):
        s = tr[tr["category"] == cat]
        aL, _ = R._score(s, "rf"); aB, bB = R._score(s, "blend")
        keep[cat] = (aB >= aL - 1e-3 + margin) and (min(bB, 0) >= min(_cat_bias(s, "rf"), 0) - 0.02)
    bev = R.apply_blend(ev, fitted) * sc
    use = ev["category"].map(keep).fillna(False).to_numpy(bool)
    e = ev.copy()
    e["blend"] = np.where(use, bev, ev["rf"].to_numpy(float))
    aL, _ = R._score(e, "rf"); aB, bB = R._score(e, "blend")
    return dict(label=f"E lasso +cal +guardrail(m={margin})", acc=aB, dacc=aB - aL, bias=bB,
                wins=wins(e), ncats=e["category"].nunique(), kept=sum(keep.values())), e


def main():
    data = C.load_all()
    train_df = pd.concat([data[s] for s in C.TRAIN], ignore_index=True)
    ev = data[C.EVAL]
    print(f"train {C.TRAIN} ({len(train_df):,} rows) -> eval {C.EVAL} ({len(ev):,} rows), GTIN grain, lags {C.LAGS}\n")

    fits = {
        "A prod(guardrail)": C.fit_guardrail_on(train_df),
        "B raw":             C.fit_guardrail_off(train_df),
        f"D lasso(a={A_LASSO})": C.fit_hier(train_df, A_LASSO, l1=True),
    }
    rows = []
    detail = ["D lasso", "category", None]      # capture per-category table for the lasso+cat-cal winner
    for lbl, f in fits.items():
        run_one(train_df, ev, f, lbl, rows, detail)

    # --- the synthesis: calibrate, THEN guardrail the calibrated blend (per-category, train-decided) ---
    lasso_fit = fits[f"D lasso(a={A_LASSO})"]
    syn_rows = []
    for m in (0.0, 0.01):
        r, _ = cal_then_guardrail(train_df, ev, lasso_fit, m)
        syn_rows.append(r)

    aL = R._score(ev.assign(blend=ev["rf"]), "blend")[0]
    print("=" * 78)
    print(f"OUT-OF-SAMPLE on {C.EVAL}    pure LEGO acc = {aL:.4f}")
    print("=" * 78)
    print(f"{'approach':<30}{'OOS acc':>9}{'Δ LEGO':>9}{'OOS bias':>10}{'cats≥LEGO':>11}")
    print("-" * 78)
    for r in rows:
        mark = "  ←" if (r["dacc"] > 0.01 and abs(r["bias"]) < 0.025) else ""
        print(f"{r['label']:<30}{r['acc']:>9.4f}{r['dacc']:>+9.4f}{r['bias']:>+10.4f}"
              f"{str(r['wins'])+'/'+str(r['ncats']):>11}{mark}")
    print("-" * 78)
    for r in syn_rows:
        mark = "  ←" if (r["dacc"] > 0.01 and abs(r["bias"]) < 0.025) else ""
        print(f"{r['label']:<30}{r['acc']:>9.4f}{r['dacc']:>+9.4f}{r['bias']:>+10.4f}"
              f"{str(r['wins'])+'/'+str(r['ncats']):>11}{mark}   (kept {r['kept']}/{r['ncats']} cats)")
    print("-" * 78)
    print("← = beats LEGO on accuracy (>+1pp) AND keeps |bias| < 2.5%  (the prize: accuracy + clean bias)")

    if detail[2] is not None:
        e = detail[2]
        print(f"\nPer-category — {'D lasso +cal:category':<24} (the calibrated winner) vs LEGO")
        print(f"{'category':<26}{'acc_LEGO':>9}{'acc_cal':>9}{'Δacc':>8}{'bias_LEGO':>11}{'bias_cal':>10}  win")
        for cat in sorted(e["category"].unique()):
            s = e[e["category"] == cat]
            aLc, bLc = R._score(s.assign(blend=s["rf"]), "blend")
            aBc, bBc = R._score(s, "blend")
            win = (aBc >= aLc - 1e-3) and (min(bBc, 0) >= min(bLc, 0) - 0.02)
            print(f"{cat:<26}{aLc:>9.3f}{aBc:>9.3f}{aBc-aLc:>+8.3f}{bLc:>+11.3f}{bBc:>+10.3f}  {'✓' if win else ''}")


if __name__ == "__main__":
    main()
