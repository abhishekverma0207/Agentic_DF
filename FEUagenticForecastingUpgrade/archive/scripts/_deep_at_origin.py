#!/usr/bin/env python3
"""Train the global TFT at an arbitrary ORIGIN (for the rolling-origin lock). Forecasts
ORIGIN+1..ORIGIN+13 leak-free (history <= ORIGIN). Saves deep_TFT_<loss>_<origin>_fc.parquet.

Usage:  python scripts/_deep_at_origin.py <origin> <loss1,loss2,...>
        python scripts/_deep_at_origin.py 202541 Huber3
"""
import sys, os
sys.path.insert(0, ".")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from utils.deep_forecast import train_global_deep

ORIGIN = int(sys.argv[1]) if len(sys.argv) > 1 else 202541
LOSSES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["Huber3"]
CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]


def load():
    fr = []
    for c in CATS:
        d = pd.read_parquet(f"sourcedata/THAILAND/by_category/{c}.parquet")
        d = d[d["key"].astype(str).str.endswith("_E1016")].copy(); d["__cat"] = c; fr.append(d)
    return pd.concat(fr, ignore_index=True)


def main():
    df = load()
    print(f"[deep@{ORIGIN}] {df['key'].nunique()} E1016 series; losses={LOSSES}", flush=True)
    for ln in LOSSES:
        fc = train_global_deep(df, key_col="key", period_col="year_week", target_col="Actuals",
                               category_col="Category", origin_period=ORIGIN, horizon=13,
                               model_name="TFT", max_steps=2000, loss_name=ln)
        out = f"artifacts_th_local/deep_TFT_{ln}_{ORIGIN}_fc.parquet"
        fc.to_parquet(out, index=False)
        print(f"[deep@{ORIGIN}] saved {out} ({len(fc)} rows)", flush=True)


if __name__ == "__main__":
    main()
