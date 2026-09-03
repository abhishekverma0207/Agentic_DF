"""Assemble artifacts_th_local/final_forecast.parquet = the best validated forecast per
category (DIQ model). SOURCE map below is the current best; swap to the clean-run files
(artifacts_th_local/final/fc_<CAT>.parquet) once that run completes for the legitimate
single-config version. Combines per-category -> [key, year_week, predicted]."""
import sys; sys.path.insert(0, ".")
import pandas as pd
from utils import lego_eval

base = lego_eval.load_eval_base()
catmap = base[["cs", "Category"]].drop_duplicates().set_index("cs")["Category"]

# per-category source (current best-available; recency=26 run2 wins FAB_ENH/HAIR/BODY,
# run3 wins HOME/SKIN/DEO via correction, per-key rescue wins FABRIC). FACE/FOODS = best global.
RUN2 = "artifacts_th_local/run2/all_forecasts.parquet"
RUN3 = "artifacts_th_local/run3/all_forecasts.parquet"
FAB = "artifacts_th_local/FABRIC_CLEANING_hybrid/_hybrid_oos_fc.parquet"
SRC = {"BODY": RUN2, "DEODORANTS & FRAGRANCES": RUN3, "FABRIC CLEANING": FAB,
       "FABRIC ENHANCERS": RUN2, "FACE": RUN2, "FOODS": RUN3, "HAIR CARE": RUN2,
       "HOME & HYGIENE": RUN3, "SKIN CLEANSING": RUN3}

cache = {}
parts = []
for cat, path in SRC.items():
    if path not in cache:
        d = pd.read_parquet(path)[["key", "year_week", "predicted"]]
        d["__cat"] = d["key"].astype(str).str[:-6].map(catmap)
        cache[path] = d
    d = cache[path]
    parts.append(d[d["__cat"] == cat][["key", "year_week", "predicted"]])
out = pd.concat(parts, ignore_index=True)
out.to_parquet("artifacts_th_local/final_forecast.parquet", index=False)
print(f"final_forecast.parquet: {len(out)} rows, {out['key'].nunique()} keys, "
      f"{out['key'].astype(str).str[:-6].map(catmap).nunique()} categories")
