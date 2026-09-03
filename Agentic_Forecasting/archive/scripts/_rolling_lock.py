#!/usr/bin/env python3
"""Rolling-origin LOCK — leak-free validation that the two NEW levers generalize beyond the 202602
test window. At each held-out origin O (last-known week; forecast O+1..O+13, all actuals known but
NO LEGO benchmark there) we ask: do the levers improve OUR OWN error vs the base, at the eval grain
(CS_BARCODE x year_week, account E1016)?

  (A) FABRIC un-damping gate:  apply the SAME gate (CV<0.8, trend>0.75, floor=1.05) to BASE(O);
      does it lower FABRIC WAPE AND |bias|?  (the lift is leak-free: recent-13 + stability as-of O)
  (B) FOODS diverse ensemble:  0.2*BASE + 0.4*deepTFT-huber3 + 0.4*LightGBM-rmse;
      does the ensemble beat EVERY single member on FOODS WAPE?  (the diversity benefit)

If both hold at 202541 AND 202548 (origins the recipe was NOT tuned on), the levers generalize and
the 202602 result is defensible (not test-window-fitted).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _undamp_gate import apply_undamp

VAL_ORIGINS = [202541, 202548]
SRC = 'sourcedata/THAILAND/by_category'
FOODS_W = dict(base=0.2, tft=0.4, lgb=0.4)


def score(f, a):
    a = np.asarray(a, float); f = np.asarray(f, float); s = np.abs(a).sum()
    return 1 - np.abs(f - a).sum() / s, (f - a).sum() / s


def src(cat):
    d = pd.read_parquet(f'{SRC}/{cat}.parquet', columns=['key', 'year_week', 'Actuals'])
    d = d[d['key'].astype(str).str.endswith('_E1016')].copy()
    d['cs'] = d['key'].astype(str).str[:-6]; d['yw'] = d['year_week'].astype(int)
    return d


def cell_actuals(cat, wk):
    d = src(cat)
    return (d[d.yw.isin(wk)].groupby(['cs', 'yw'])['Actuals'].sum()
            .rename('actual').reset_index().rename(columns={'yw': 'period'}))


def cell_fc(path, wk):
    if not os.path.exists(path):
        return None
    fc = pd.read_parquet(path); fc['cs'] = fc['key'].astype(str).str[:-6]; fc['period'] = fc['year_week'].astype(int)
    fc = fc[fc.period.isin(wk)]
    return fc.groupby(['cs', 'period'])['predicted'].sum().rename('predicted').reset_index()


def fcst_weeks(O, n=13):
    wk = sorted(w for w in src('FOODS').yw.unique() if w > O)
    return wk[:n]


def main():
    print('=' * 82)
    print('ROLLING-ORIGIN LOCK — leak-free generalization of the un-damp + FOODS-ensemble levers')
    print('  (metric = our own WAPE-accuracy & bias at the CS_BARCODE x week grain; no LEGO at past origins)')
    print('=' * 82)
    fab_ok, foods_ok = [], []
    for O in VAL_ORIGINS:
        wk = fcst_weeks(O)
        if len(wk) < 13:
            print(f'origin {O}: only {len(wk)} future weeks -> skip'); continue
        print(f'\n--- origin {O}: forecast {wk[0]}..{wk[-1]} ---')

        # ===== (A) FABRIC un-damp =====
        bpath = f'/tmp/base_{O}.parquet'
        bcells = cell_fc(bpath, wk)
        if bcells is not None:
            af = cell_actuals('FABRIC_CLEANING', wk)
            df = af.merge(bcells, on=['cs', 'period'], how='left').fillna({'predicted': 0.0})
            lift, n = apply_undamp(df[['cs', 'period', 'predicted']], 'FABRIC CLEANING', 'FABRIC_CLEANING',
                                   cv=0.8, tr=0.75, floor=1.05, origin=wk[0])
            df['gated'] = df['predicted'].values * df['cs'].map(lift).fillna(1.0).values
            a0, b0 = score(df['predicted'], df['actual']); ag, bg = score(df['gated'], df['actual'])
            both = ag > a0 and abs(bg) < abs(b0); fab_ok.append(both)
            tag = 'IMPROVES BOTH' if both else (('acc+ ' if ag > a0 else '') + ('|bias|-' if abs(bg) < abs(b0) else '') or 'no improvement')
            print(f'  [A] FABRIC: base acc {a0:.4f} bias {b0:+.4f}  ->  un-damped acc {ag:.4f} bias {bg:+.4f}  (lifted {n} keys)  {tag}')
        else:
            print(f'  [A] base_{O}.parquet not ready')

        # ===== (B) FOODS ensemble =====
        members = {'base': f'/tmp/base_{O}.parquet', 'tft': f'artifacts_th_local/deep_TFT_Huber3_{O}_fc.parquet',
                   'lgb': f'/tmp/lgb_{O}.parquet'}
        mc = {k: cell_fc(p, wk) for k, p in members.items()}
        if all(c is not None for c in mc.values()):
            af = cell_actuals('FOODS', wk)
            df = af.copy()
            for k in mc:
                df = df.merge(mc[k].rename(columns={'predicted': k}), on=['cs', 'period'], how='left')
            df = df.fillna(0.0)
            df['ens'] = FOODS_W['base'] * df['base'] + FOODS_W['tft'] * df['tft'] + FOODS_W['lgb'] * df['lgb']
            scm = {k: score(df[k], df['actual']) for k in ['base', 'tft', 'lgb']}
            se = score(df['ens'], df['actual'])
            beat = se[0] >= max(scm[k][0] for k in scm) - 1e-9; foods_ok.append(beat)
            mem = '  '.join(f'{k} {scm[k][0]:.4f}/{scm[k][1]:+.3f}' for k in scm)
            print(f'  [B] FOODS members (acc/bias): {mem}')
            print(f'      ENSEMBLE acc {se[0]:.4f} bias {se[1]:+.4f}   {"BEATS every member" if beat else "below best member ("+f"{max(scm[k][0] for k in scm):.4f})"}')
        else:
            print(f'  [B] FOODS waiting on: {[k for k,c in mc.items() if c is None]}')

    print('\n' + '=' * 82)
    if fab_ok:
        print(f'(A) FABRIC un-damp improves BOTH acc & |bias| at ALL {len(fab_ok)} held-out origins: {all(fab_ok)}')
    if foods_ok:
        print(f'(B) FOODS ensemble >= best single member at ALL {len(foods_ok)} held-out origins: {all(foods_ok)}')
    print('=' * 82)


if __name__ == '__main__':
    main()
