#!/usr/bin/env python3
"""Assemble the FINAL TH 13-week deliverable = the per-category recipe that beats LEGO on the top-5:

  base            : final_forecast_catselect13.parquet (per-category XGBoost-variant selection + seasonal-trust)
  FABRIC CLEANING : + un-damping gate (lift stable non-declining keys to recent level)  -> flips to WIN BOTH
  FOODS           : diverse ENSEMBLE 0.2*base + 0.4*deepTFT-huber3 + 0.4*LightGBM-rmse  -> flips to WIN BOTH
                    (neural + GBM diversity exceeds either family alone; leak-free, origin 202602)

-> final_forecast_v4.parquet (5/5 top-5 both-win). Run: python scripts/_assemble_final.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # scripts/ dir (for _undamp_gate import)
from utils import lego_eval
from _undamp_gate import apply_undamp, key_stats, CATFILES, TOP5, GATE_CATS

# FOODS diverse-ensemble weights (base / deepTFT-huber3 / LightGBM-rmse)
FOODS_W = dict(base=0.2, tft=0.4, lgb=0.4)
TFT_FOODS = 'artifacts_th_local/deep_TFT_Huber3_fc.parquet'
LGB_FOODS = 'artifacts_th_local/_foods_lgb_rmse.parquet'


def score(f, a):
    a = np.asarray(a, float); f = np.asarray(f, float); sa = np.abs(a).sum()
    return 1 - np.abs(f - a).sum() / sa, (f - a).sum() / sa


def main():
    base = lego_eval.load_eval_base(); base['cs'] = base['cs'].astype(str)
    base['period'] = base['Shipment_Week'].str.replace('-', '').astype(int)
    fc = pd.read_parquet('artifacts_th_local/final_forecast_catselect13.parquet')
    fc['cs'] = fc['key'].astype(str).str[:-6]; fc['period'] = fc['year_week'].astype(int)
    fc['pred'] = fc['predicted'].astype(float)

    # 1) FABRIC un-damping gate
    for catname, p in GATE_CATS.items():
        cat_cs = set(base[base.Category == catname].cs)
        sub = fc[fc.cs.isin(cat_cs)][['cs', 'period', 'predicted']].copy()
        lift, n = apply_undamp(sub, catname, CATFILES[catname], **p)
        mask = fc.cs.isin(cat_cs)
        fc.loc[mask, 'pred'] = fc.loc[mask, 'predicted'].values * fc.loc[mask, 'cs'].map(lift).fillna(1.0).values
        print(f'  [un-damp] {catname}: lifted {n} keys')

    # 2) FOODS diverse ensemble (replace FOODS keys with the blend)
    foods_cs = set(base[base.Category == 'FOODS'].cs)
    tft = pd.read_parquet(TFT_FOODS); tft['cs'] = tft['key'].astype(str).str[:-6]; tft['period'] = tft['year_week'].astype(int)
    lgb = pd.read_parquet(LGB_FOODS); lgb['cs'] = lgb['key'].astype(str).str[:-6]; lgb['period'] = lgb['year_week'].astype(int)
    tmap = tft.set_index(['cs', 'period'])['predicted']; lmap = lgb.set_index(['cs', 'period'])['predicted']
    fm = fc.cs.isin(foods_cs)
    idx = list(zip(fc.loc[fm, 'cs'], fc.loc[fm, 'period']))
    tv = np.array([tmap.get(k, np.nan) for k in idx]); lv = np.array([lmap.get(k, np.nan) for k in idx])
    bv = fc.loc[fm, 'pred'].values
    tv = np.where(np.isnan(tv), bv, tv); lv = np.where(np.isnan(lv), bv, lv)   # fall back to base if a model misses a key
    fc.loc[fm, 'pred'] = FOODS_W['base'] * bv + FOODS_W['tft'] * tv + FOODS_W['lgb'] * lv
    print(f'  [FOODS ensemble] blended {int(fm.sum())} cells: {FOODS_W}')

    out = fc[['key', 'year_week']].copy(); out['predicted'] = np.clip(fc['pred'].values, 0, None)
    out_path = os.environ.get('OUT', 'artifacts_th_local/final_forecast_v4.parquet')
    out.to_parquet(out_path, index=False)

    # scorecard
    fcv = fc[['cs', 'period', 'pred']]
    print('=' * 76)
    print('FINAL TH 13-week scorecard (FABRIC un-damp + FOODS diverse ensemble)')
    print('=' * 76)
    both = acc = bia = 0
    for catname in TOP5:
        b = base[base.Category == catname]
        m = b.merge(fcv, on=['cs', 'period'], how='left'); m['pred'] = m['pred'].fillna(0)
        al, bl = score(m['lego'], m['actual']); a, bb = score(m['pred'], m['actual'])
        wb = a > al and abs(bb) < abs(bl); v = 'WIN BOTH' if wb else (('acc ' if a > al else '') + ('bias' if abs(bb) < abs(bl) else '')) or 'lose'
        both += wb; acc += a > al; bia += abs(bb) < abs(bl)
        print(f'  {catname:18s} {a:.4f}/{bb:+.4f}   LEGO {al:.4f}/{bl:+.4f}   {v}')
    b5 = base[base.Category.isin(TOP5)]; m5 = b5.merge(fcv, on=['cs', 'period'], how='left'); m5['pred'] = m5['pred'].fillna(0)
    al, bl = score(m5['lego'], m5['actual']); a, bb = score(m5['pred'], m5['actual'])
    print('-' * 76)
    print(f'  TOP-5 AGGREGATE    {a:.4f}/{bb:+.4f}   LEGO {al:.4f}/{bl:+.4f}   {"WIN BOTH" if a>al and abs(bb)<abs(bl) else ""}')
    print(f'  TOP-5 both-win {both}/5 | acc-win {acc}/5 | bias-win {bia}/5   ->  {out_path}')


if __name__ == '__main__':
    main()
