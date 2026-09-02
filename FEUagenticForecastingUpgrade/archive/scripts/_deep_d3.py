"""Step D-3 (the deep push): retrain the global TFT with forecastability features as STATIC
covariates (ADI, CV2, spike, recent-level, volume, nonzero-frac, history-length) so the global
model can DIFFERENTIATE series the way LEGO's per-key RF does — targeting the 13wk accuracy gap on
FABRIC CLEANING / HAIR CARE / FOODS. Loss = Huber8 (best raw 13wk accuracy). Saves
deep_TFT_d3_fc.parquet and prints a per-category 13wk eval vs the current Huber8 and vs LEGO."""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from utils.deep_forecast import train_global_deep

ORIGIN = int(sys.argv[1]) if len(sys.argv) > 1 else 202602
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
LOSS = sys.argv[3] if len(sys.argv) > 3 else "Huber8"
OUT = "artifacts_th_local/deep_TFT_d3_fc.parquet"
CATS = ["BODY", "DEODORANTS_AND_FRAGRANCES", "FABRIC_CLEANING", "FABRIC_ENHANCERS", "FACE",
        "FOODS", "HAIR_CARE", "HOME_AND_HYGIENE", "SKIN_CLEANSING"]

# forecastability features -> standardized static covariates
F = pd.read_parquet("artifacts_th_local/_key_features.parquet"); F["key"] = F["key"].astype(str); F = F.set_index("key")
sx = pd.DataFrame(index=F.index)
sx["log_vol"] = np.log1p(F["vol"]); sx["log_mean"] = np.log1p(F["mean"]); sx["nzfrac"] = F["nzfrac"]
sx["log_adi"] = np.log1p(F["ADI"]); sx["log_cv2"] = np.log1p(F["CV2"]); sx["spike"] = F["spike"]
sx["recent"] = F["recent"]; sx["log_n"] = np.log1p(F["n"])
sx = sx.replace([np.inf, -np.inf], np.nan)
sx = (sx - sx.mean()) / sx.std(ddof=0).replace(0, 1.0)
sx = sx.fillna(0.0)
print(f"[d3] static covariates: {list(sx.columns)} over {len(sx)} keys", flush=True)

fr = []
for c in CATS:
    d = pd.read_parquet(f"sourcedata/THAILAND/by_category/{c}.parquet")
    d = d[d["key"].astype(str).str.endswith("_E1016")].copy(); fr.append(d)
raw = pd.concat(fr, ignore_index=True)
print(f"[d3] training global TFT ({LOSS}, +{len(sx.columns)} static covs) on {raw['key'].nunique()} E1016 keys", flush=True)

deep = train_global_deep(raw, key_col="key", period_col="year_week", target_col="Actuals",
                         category_col="Category", origin_period=ORIGIN, horizon=13,
                         model_name="TFT", max_steps=STEPS, loss_name=LOSS, extra_static_df=sx)
deep[["key", "year_week", "predicted"]].to_parquet(OUT, index=False)
print(f"[d3] saved {OUT}: {deep.shape}", flush=True)

# quick per-category 13wk eval vs current Huber8 + LEGO
from utils import lego_eval
base = lego_eval.load_eval_base(); base["cs"] = base["cs"].astype(str)
CATSN = sorted(c for c in base.Category.unique() if c != "ICE CREAM")
H = pd.read_parquet("artifacts_th_local/deep_TFT_Huber8_fc.parquet"); H["key"] = H["key"].astype(str)


def ab(pc):
    a = pc.actual.values; f = pc.ours.fillna(0).values; tot = np.abs(a).sum()
    return (1 - np.abs(f - a).sum() / max(tot, 1e-9), (f - a).sum() / max(tot, 1e-9))


def per_cat(fc, name):
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted"); p = p[p.Category != "ICE CREAM"]
    print(f"--- {name} (13wk) ---", flush=True)
    for c in CATSN:
        pc = p[p.Category == c]; a, b = ab(pc); la, lb = ab(pc.assign(ours=pc.lego))
        tag = " *" if c in {"FABRIC CLEANING", "HAIR CARE", "FOODS"} else ""
        print(f"  {c:24s} acc {a:+.3f}/{la:+.3f}  bias {b:+.3f}/{lb:+.3f}{tag}", flush=True)


per_cat(H.rename(columns={"predicted": "predicted"}), "Huber8 (current)")
per_cat(deep, "Huber8 + D3 static covariates")
