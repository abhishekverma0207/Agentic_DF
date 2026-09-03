#!/usr/bin/env python
"""Does a REGULARIZED meta-learner generalise better than per-segment fit + guardrail?

Apples-to-apples on the cached DE components (reuses de_segment_research's metrics +
apply_blend). Four stackers, all fit on 202601/606/611, scored OUT-OF-SAMPLE on 202616
at GTIN grain, lags 4/5:

  A guardrail ON   (prod)  — per-(cat×seg) WAPE-asym fit, keep stack only if it beats LEGO
                             by margin, else [0,0,1]; + hierarchical shrink to category.
  B guardrail OFF  (raw)   — per-(cat×seg) WAPE-asym fit, no guardrail, no shrink.
  C ridge-hier             — global→category→segment, each level's weights shrunk toward
                             its parent by an L2 penalty; penalty strength α picked OUT-OF-FOLD.
  D lasso-hier             — same, but L1 penalty toward the parent (sparse deviations).

Run:  .venv/bin/python scripts/de_stacker_compare.py
"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.de_segment_research as R          # reuse metrics, apply_blend, constants
from utils.weight_fit import fit_per_category_weights

NB = int(os.environ.get("DE_SEG_NBOOST", "300"))
CACHE = Path("datacache/de/parallel_run")
TRAIN = ["202601", "202606", "202611"]
EVAL = "202616"
LAGS = [4, 5]
WEEK = R.WEEK


# --------------------------------------------------------------------------- #
# Data: cached components + realized actuals, lags 4/5
# --------------------------------------------------------------------------- #
def load(snap):
    c = pd.read_parquet(CACHE / "components" / f"comp_{snap}_h{LAGS[0]}-{LAGS[-1]}_nb{NB}.parquet")
    return c[c["horizon"].isin(LAGS)]


def load_all():
    acts = pd.read_parquet(CACHE / "actuals" / "actuals_202625.parquet")
    acts["key"] = acts["key"].astype(str)
    acts[WEEK] = acts[WEEK].astype(str).str.replace("-", "", regex=False)
    acts = acts.rename(columns={"Actuals": "actual"})[["key", WEEK, "actual"]] \
        .groupby(["key", WEEK], as_index=False)["actual"].sum()
    out = {}
    for s in TRAIN + [EVAL]:
        c = load(s).merge(acts, on=["key", WEEK], how="inner")
        out[s] = c
    return out


# --------------------------------------------------------------------------- #
# A — guardrail ON (prod): identical to de_segment_research.fit_segments
# B — guardrail OFF (raw)
# --------------------------------------------------------------------------- #
def fit_guardrail_on(df):
    return R.fit_segments(df, shrink=True)


def fit_guardrail_off(df):
    d = df.assign(g=df["s"], l=df["c"], catseg=df["category"] + " | " + df["segment"])
    seg = fit_per_category_weights(d, cat_col="catseg", grain="gtin", objective="wape_asym",
                                   bias_tolerance=0.02, guardrail=False, lego_col="rf",
                                   gtin_col="cs_gtin", week_col=WEEK)
    return {k: {"weights": v["weights"], "use": "RAW"} for k, v in seg.items()}


# --------------------------------------------------------------------------- #
# C / D — hierarchical regularized stacker (global -> category -> segment)
# --------------------------------------------------------------------------- #
def _agg(sub):
    g = sub.groupby(["cs_gtin", WEEK], as_index=False).agg(
        s=("s", "sum"), c=("c", "sum"), rf=("rf", "sum"), a=("actual", "sum"))
    return g[["s", "c", "rf"]].to_numpy(float), g["a"].to_numpy(float)


def _wape(X, w, A):
    return np.abs(X @ w - A).sum() / max(A.sum(), 1e-9)


def _fit_pen(X, A, prior, alpha, l1):
    """Minimise the SAME asymmetric-WAPE objective as prod (WAPE + under-forecast penalty
    vs LEGO) + a ridge(L2)/lasso(L1) penalty toward the parent `prior`."""
    prior = np.asarray(prior, float)
    tot = max(A.sum(), 1e-9)
    floor = min((X[:, 2] - A).sum() / tot, 0.0) - 0.02      # don't under-forecast worse than LEGO − tol
    def obj(w):
        pred = X @ w
        wape = np.abs(pred - A).sum() / tot
        bM = (pred - A).sum() / tot
        under = max(0.0, floor - min(bM, 0.0))
        pen = np.abs(w - prior).sum() if l1 else float(((w - prior) ** 2).sum())
        return wape + 200.0 * under ** 2 + alpha * pen
    best = min((minimize(obj, s0, method="Nelder-Mead", bounds=[(0, 3)] * 3,
                         options={"maxiter": 800, "xatol": 1e-3, "fatol": 1e-8})
                for s0 in ([0, 0, 1.0], list(prior), [0.3, 0.3, 0.4])), key=lambda r: r.fun)
    return [round(float(v), 4) for v in best.x]


def fit_hier(df, alpha, l1=False):
    Xg, Ag = _agg(df)
    wg = _fit_pen(Xg, Ag, [0, 0, 1.0], 0.0, l1)               # global blend (unpenalised)
    out = {}
    for cat, csub in df.groupby("category"):
        Xc, Ac = _agg(csub)
        wc = _fit_pen(Xc, Ac, wg, alpha, l1)                  # category shrinks toward global
        for seg, ssub in csub.groupby("segment"):
            Xs, As = _agg(ssub)
            ws = _fit_pen(Xs, As, wc, alpha, l1) if As.sum() > 0 else wc   # segment shrinks toward category
            if sum(ws) < 0.1:        # degenerate all-zero fit (near-zero-volume segment) → defer to category
                ws = wc
            out[f"{cat} | {seg}"] = {"weights": ws, "use": "REG"}
    return out


def pick_alpha(train, l1, grid=(0.0, 0.05, 0.2, 0.5, 1.5)):
    """Leave-one-train-snapshot-out: pick α minimising mean held-out WAPE."""
    snaps = list(train.keys())
    best_a, best_w = grid[0], 1e9
    for a in grid:
        woof = []
        for held in snaps:
            fitted = fit_hier(pd.concat([train[s] for s in snaps if s != held], ignore_index=True), a, l1)
            ev = train[held].copy()
            ev["blend"] = R.apply_blend(ev, fitted)
            aB, _ = R._score(ev, "blend")
            woof.append(1.0 - aB)
        m = float(np.mean(woof))
        print(f"    α={a:<5} mean OOF WAPE={m:.4f}")
        if m < best_w:
            best_w, best_a = m, a
    return best_a


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
def evaluate(ev, fitted, label, insample_acc):
    e = ev.copy()
    e["blend"] = R.apply_blend(e, fitted)
    aL, bL = R._score(e, "rf")
    aB, bB = R._score(e, "blend")
    wins = 0
    for cat in sorted(e["category"].unique()):
        s = e[e["category"] == cat]
        al, _ = R._score(s, "rf"); ab, bb = R._score(s, "blend")
        if (ab >= al - 1e-3) and (min(bb, 0) >= min(_score_bias(s, "rf"), 0) - 0.02):
            wins += 1
    ncats = e["category"].nunique()
    nlego = sum(1 for v in fitted.values()
                if v["weights"][2] >= 0.9 and v["weights"][0] < 0.1 and v["weights"][1] < 0.1)
    return dict(label=label, acc_lego=aL, acc=aB, dacc=aB - aL, bias=bB,
                wins=wins, ncats=ncats, nlego=nlego, ngrp=len(fitted),
                insample=insample_acc, overfit=insample_acc - aB)


def _score_bias(s, col):
    g = s.groupby(["cs_gtin", WEEK]).agg(p=(col, "sum"), a=("actual", "sum")).reset_index()
    return R._bias(g.p, g.a)


def insample(train_df, fitted):
    e = train_df.copy()
    e["blend"] = R.apply_blend(e, fitted)
    a, _ = R._score(e, "blend")
    return a


def main():
    print(f"Loading cached components (nb{NB}, lags {LAGS}) …")
    data = load_all()
    train_df = pd.concat([data[s] for s in TRAIN], ignore_index=True)
    ev = data[EVAL]
    print(f"train {TRAIN}: {len(train_df):,} rows  |  eval {EVAL}: {len(ev):,} rows\n")

    print("Picking α out-of-fold for the regularized stackers …")
    print("  ridge:")
    a_ridge = pick_alpha({s: data[s] for s in TRAIN}, l1=False)
    print("  lasso:")
    a_lasso = pick_alpha({s: data[s] for s in TRAIN}, l1=True)
    print(f"  -> ridge α={a_ridge}, lasso α={a_lasso}\n")

    fits = {
        "A guardrail-ON (prod)":  fit_guardrail_on(train_df),
        "B guardrail-OFF (raw)":  fit_guardrail_off(train_df),
        f"C ridge-hier (α={a_ridge})":  fit_hier(train_df, a_ridge, l1=False),
        f"D lasso-hier (α={a_lasso})":  fit_hier(train_df, a_lasso, l1=True),
    }

    rows = [evaluate(ev, f, lbl, insample(train_df, f)) for lbl, f in fits.items()]

    aL = rows[0]["acc_lego"]
    print("=" * 96)
    print(f"OUT-OF-SAMPLE on {EVAL} (GTIN grain, lags {LAGS})   |   pure LEGO acc = {aL:.4f}")
    print("=" * 96)
    print(f"{'approach':<26}{'OOS acc':>9}{'Δ vs LEGO':>11}{'OOS bias':>10}{'cats≥LEGO':>11}"
          f"{'in-samp':>9}{'overfit':>9}{'≈LEGO segs':>12}")
    print("-" * 96)
    for r in rows:
        print(f"{r['label']:<26}{r['acc']:>9.4f}{r['dacc']:>+11.4f}{r['bias']:>+10.4f}"
              f"{str(r['wins'])+'/'+str(r['ncats']):>11}{r['insample']:>9.4f}{r['overfit']:>+9.4f}"
              f"{str(r['nlego'])+'/'+str(r['ngrp']):>12}")
    print("-" * 96)
    print("Δ vs LEGO > 0 = beats LEGO out-of-sample. 'overfit' = in-sample − OOS (smaller = generalises better).")
    print("'≈LEGO segs' = groups that collapsed to ~pure LEGO (w_rf≥0.9).")


if __name__ == "__main__":
    main()
