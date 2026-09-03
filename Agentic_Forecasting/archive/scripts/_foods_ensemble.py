#!/usr/bin/env python3
"""FOODS ensemble ceiling-check: can ANY convex blend of the diverse bases beat LEGO on accuracy
(0.5763) while keeping |bias| < 0.0975? If the best achievable accuracy < LEGO, FOODS is confirmed
frontier-limited. Searches the convex hull via (a) all pairwise/triple blends on a grid and
(b) a constrained least-squares fit of weights to actual (the accuracy-optimal convex point)."""
import os, sys, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import lego_eval

CAND = {
    'catsel': 'artifacts_th_local/final_forecast_catselect13.parquet',
    'xgb_hb': '/tmp/hb.parquet', 'xgb_tw': '/tmp/tw.parquet', 'xgb_rp': '/tmp/rp.parquet',
    'xgb_rmse': '/tmp/f_xgb_rmse.parquet', 'xgb_pk': '/tmp/f_xgb_pk.parquet',
    'lgb_tw': '/tmp/f_lgb_tw.parquet', 'lgb_rmse': '/tmp/f_lgb_rmse.parquet',
    'tft_hb8': 'artifacts_th_local/deep_TFT_Huber8_fc.parquet',
    'tft_hb3': 'artifacts_th_local/deep_TFT_Huber3_fc.parquet',
    'tft_hb': 'artifacts_th_local/deep_TFT_Huber_fc.parquet',
    'tft_rmse': 'artifacts_th_local/deep_TFT_RMSE_fc.parquet',
    'tft_mae': 'artifacts_th_local/deep_TFT_MAE_fc.parquet',
}


def score(f, a):
    a = np.asarray(a, float); f = np.asarray(f, float); sa = np.abs(a).sum()
    return 1 - np.abs(f - a).sum() / sa, (f - a).sum() / sa


def main():
    base = lego_eval.load_eval_base(); base['cs'] = base['cs'].astype(str)
    base['period'] = base['Shipment_Week'].str.replace('-', '').astype(int)
    b = base[base.Category == 'FOODS'].copy().reset_index(drop=True)
    al, bl = score(b['lego'], b['actual'])
    A = b['actual'].values
    print(f'FOODS LEGO: acc {al:.4f} bias {bl:+.4f}   (target: acc>{al:.4f} & |bias|<{abs(bl):.4f})')

    P = {}
    for nm, fn in CAND.items():
        if not os.path.exists(fn):
            continue
        fc = pd.read_parquet(fn); fc['cs'] = fc['key'].astype(str).str[:-6]; fc['period'] = fc['year_week'].astype(int)
        m = b.merge(fc[['cs', 'period', 'predicted']], on=['cs', 'period'], how='left')
        P[nm] = m['predicted'].fillna(0).values
        a, bb = score(P[nm], A)
        print(f'  {nm:9s}: acc {a:.4f} bias {bb:+.4f}')
    names = list(P)
    M = np.column_stack([P[n] for n in names])

    # (a) accuracy-optimal convex blend via constrained NNLS-style search (minimize WAPE = L1)
    from scipy.optimize import minimize
    def negacc(w):
        w = np.clip(w, 0, None); s = w.sum()
        if s <= 0: return 1.0
        f = M @ (w / s); return np.abs(f - A).sum() / np.abs(A).sum()
    best = None
    for seed in range(12):
        w0 = np.ones(len(names)) if seed == 0 else np.abs(np.cos(np.arange(len(names)) + seed))
        r = minimize(negacc, w0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-4, fatol=1e-5))
        w = np.clip(r.x, 0, None); w = w / max(w.sum(), 1e-9)
        f = M @ w; a, bb = score(f, A)
        if best is None or a > best[0]:
            best = (a, bb, w)
    a, bb, w = best
    print(f'\n  ACCURACY-OPTIMAL convex blend: acc {a:.4f} bias {bb:+.4f}   {"WIN BOTH" if a>al and abs(bb)<abs(bl) else ("beats LEGO acc!" if a>al else "below LEGO acc")}')
    print('     weights: ' + ', '.join(f'{n}={wi:.2f}' for n, wi in zip(names, w) if wi > 0.01))

    # (b) best simple blend that satisfies the bias constraint, maximizing accuracy
    bestc = None
    for r in (2, 3):
        for combo in itertools.combinations(range(len(names)), r):
            for ws in itertools.product(np.arange(0, 1.01, 0.2), repeat=r):
                s = sum(ws)
                if abs(s - 1) > 1e-6: continue
                f = sum(ws[i] * M[:, combo[i]] for i in range(r)); a, bb = score(f, A)
                if abs(bb) < abs(bl) and (bestc is None or a > bestc[0]):
                    bestc = (a, bb, [(names[combo[i]], ws[i]) for i in range(r) if ws[i] > 0])
    if bestc:
        a, bb, ww = bestc
        print(f'  BEST blend s.t. |bias|<LEGO: acc {a:.4f} bias {bb:+.4f}   {"WIN BOTH" if a>al else "loses acc"}  [{ww}]')


if __name__ == '__main__':
    main()
