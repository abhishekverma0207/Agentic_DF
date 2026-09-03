"""Step 1 of the router: train the pooled deep model at the backtest origin (202541 ->
predict the held-out window 202542-202602) and SAVE it. Kept in its OWN process so torch's
OpenMP runtime never coexists with LightGBM's in the per-key step (that combo crashes).
Usage: _router_deep.py [origin] [max_steps]"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from utils.deep_forecast import train_global_deep

ORIGIN = int(sys.argv[1]) if len(sys.argv) > 1 else 202541
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
OUT = f"artifacts_th_local/_router_deep_{ORIGIN}.parquet"
CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]

fr = []
for c in CATS:
    d = pd.read_parquet(f"sourcedata/THAILAND/by_category/{c}.parquet")
    d = d[d["key"].astype(str).str.endswith("_E1016")].copy()
    fr.append(d)
raw = pd.concat(fr, ignore_index=True)
deep = train_global_deep(raw, key_col="key", period_col="year_week", target_col="Actuals",
                         category_col="Category", origin_period=ORIGIN, horizon=13,
                         model_name="TFT", max_steps=STEPS, loss_name="Huber8")
deep[["key", "year_week", "predicted"]].to_parquet(OUT, index=False)
print(f"[deep] saved {OUT}: {deep.shape}", flush=True)
