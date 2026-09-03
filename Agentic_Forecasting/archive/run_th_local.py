"""Local Thailand runner: run the LLM-free deterministic pipeline per category,
collect forecasts, and score them against LEGO with utils.lego_eval.

Models ALL accounts per category; LEGO (E1016) is the eval yardstick.

IMPORTANT: the pipeline uses multiprocessing 'spawn' (macOS default) which re-imports
this module in workers. ALL execution is under `if __name__ == '__main__'`.

Examples:
  # fast baseline across all 9 categories, final origin (cutoff 2026-02), score vs LEGO
  .venv/bin/python run_th_local.py --mode fast
  # full model on the 4 big-sales categories
  .venv/bin/python run_th_local.py --mode full --categories FABRIC_CLEANING HOME_AND_HYGIENE HAIR_CARE FOODS
  # shrink big categories for fast iteration (keep ALL E1016 keys, 10% of others)
  .venv/bin/python run_th_local.py --mode fast --nonE1016-frac 0.1
"""
import os
for k, v in {
    "MLFLOW_AUTOLOGGING_DISABLE": "1", "OTEL_SDK_DISABLED": "true",
    "CREWAI_DISABLE_TELEMETRY": "true", "CREWAI_DO_NOT_TRACK": "1",
    "CREWAI_TELEMETRY_OPT_OUT": "true", "LITELLM_TELEMETRY": "False",
    "POSTHOG_DISABLED": "true", "DIQ_RUNNER_SKIP_INSTALL": "1",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(k, v)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import hashlib
import time
import traceback
import pandas as pd

from config.schema import load_config_from_yaml
from utils.deterministic_pipeline import run_full_deterministic_pipeline
from utils import lego_eval

DATA_DIR = "sourcedata/THAILAND/by_category"
BASE_CFG = "config/config_th_base_local.yaml"

# Ordered by E1016 eval-sales (big first).
CATEGORIES = [
    "FABRIC_CLEANING", "HOME_AND_HYGIENE", "HAIR_CARE", "FOODS",
    "FABRIC_ENHANCERS", "BODY", "SKIN_CLEANSING", "FACE",
    "DEODORANTS_AND_FRAGRANCES",
]

# Rolling origins. 'final' = the headline (cutoff 2026-02 -> forecast 2026-03..15,
# scored vs LEGO). 'tune' = an EARLIER origin to tune knobs without peeking at the
# final test window.
ORIGINS = {
    "final": dict(train_end="202541", val_start="202542", val_end="202602",
                  test_start="202603", test_end="202615"),
    "tune":  dict(train_end="202533", val_start="202534", val_end="202546",
                  test_start="202547", test_end="202607"),
}


def _keep_key(k: str, frac: float) -> bool:
    if frac >= 1.0 or (isinstance(k, str) and k.endswith("_E1016")):
        return True
    if not isinstance(k, str):
        return False
    return (int(hashlib.md5(k.encode()).hexdigest()[:8], 16) % 1000) < int(frac * 1000)


def _maybe_sample(parquet_path: str, frac: float, out_root: str, cat: str) -> str:
    if frac >= 1.0:
        return parquet_path
    df = pd.read_parquet(parquet_path)
    df = df[df["key"].map(lambda k: _keep_key(k, frac))]
    tmp = f"{out_root}/_sampled/{cat}.parquet"
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    df.to_parquet(tmp, index=False)
    return tmp


def apply_mode(cfg, mode: str):
    if mode == "fast":
        cfg.design.dmh_n_seeds = 1
        cfg.design.dmh_objective = "tweedie"
        cfg.design.dmh_top_k_features = 40
    # 'full' keeps the base-config values (3 seeds, dmh_objective='auto', top_k=50).


def run_category(cat, mode, origin, workers, frac, out_root, seasonal=True):
    src = f"{DATA_DIR}/{cat}.parquet"
    if not os.path.exists(src):
        print(f"  [skip] {src} missing", flush=True)
        return None
    cfg = load_config_from_yaml(BASE_CFG)
    cfg.input_data_path = _maybe_sample(src, frac, out_root, cat)
    art = f"{out_root}/{cat}"
    cfg.artifact_base_path = art
    cfg.output_folder_path = art
    o = ORIGINS[origin]
    for k, v in o.items():
        setattr(cfg, k, v)
    cfg.design.parallel_training_workers = workers
    apply_mode(cfg, mode)
    run_full_deterministic_pipeline(cfg, clean_artifacts=True)
    cands = (glob.glob(f"{art}/**/inference_forecast.parquet", recursive=True) +
             glob.glob(f"{art}/**/inference_forecast.csv", recursive=True))
    if not cands:
        return None
    f = cands[0]
    df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
    # Seasonal-level correction (the proven lever vs LEGO): scale each key's forecast
    # so its forecast-period total matches its SPLY (same-weeks-last-year) total.
    if seasonal:
        from utils import seasonal_blend
        hist = pd.read_parquet(src, columns=["key", "year_week", "Actuals", "Category"])
        if "Category" not in df.columns:
            df["Category"] = df["key"].map(
                hist.drop_duplicates("key").set_index("key")["Category"])
        df["predicted_raw"] = df["predicted"]
        df = seasonal_blend.seasonal_correct(df, hist, level="adaptive")
        print(f"  [seasonal] applied per-key SPLY scale to {cat}", flush=True)
    df["__category"] = cat
    return df


def main():
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                        format="%(asctime)s %(levelname).1s %(name)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--origin", choices=["final", "tune"], default="final")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--nonE1016-frac", type=float, default=1.0)
    ap.add_argument("--out-root", default="artifacts_th_local/run")
    ap.add_argument("--pred-col", default="predicted")
    ap.add_argument("--no-seasonal-correct", action="store_true",
                    help="disable the SPLY seasonal-level correction (for ablation)")
    a = ap.parse_args()
    os.makedirs(a.out_root, exist_ok=True)

    forecasts = []
    timings = {}
    for cat in a.categories:
        print(f"\n===== {cat}  ({a.mode}/{a.origin}, frac={a.nonE1016_frac}) =====", flush=True)
        t0 = time.time()
        try:
            df = run_category(cat, a.mode, a.origin, a.workers, a.nonE1016_frac, a.out_root,
                              seasonal=not a.no_seasonal_correct)
            dt = time.time() - t0
            timings[cat] = dt
            if df is not None:
                forecasts.append(df)
                print(f"  [ok] {cat}: {df.shape} in {dt:.0f}s", flush=True)
            else:
                print(f"  [warn] {cat}: no forecast produced ({dt:.0f}s)", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"  [FAIL] {cat}: {e}", flush=True)

    if not forecasts:
        print("\nNo forecasts produced — aborting eval.", flush=True)
        return
    allfc = pd.concat(forecasts, ignore_index=True)
    allfc.to_parquet(f"{a.out_root}/all_forecasts.parquet", index=False)
    print(f"\nTotal forecast rows: {len(allfc):,}  | timings(s): "
          f"{ {k: round(v) for k, v in timings.items()} }", flush=True)

    if a.origin == "final":
        sb, _ = lego_eval.evaluate(allfc, pred_col=a.pred_col, label=f"Ours[{a.mode}]")
        sb.to_csv(f"{a.out_root}/scoreboard.csv", index=False)
        print(f"\nScoreboard saved -> {a.out_root}/scoreboard.csv", flush=True)
    else:
        print("\n[tune origin] vs-LEGO scoring is only for 'final'; inspect "
              "all_forecasts.parquet against actuals for tuning.", flush=True)


if __name__ == "__main__":
    main()
