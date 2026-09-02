"""Smoke test: run the LLM-free deterministic pipeline locally on the smallest TH
category (HOME & HYGIENE) with FAST DMH settings, to validate the path end-to-end.

NOTE: the pipeline uses multiprocessing with the 'spawn' start method (macOS default),
which RE-IMPORTS this module in every worker. All execution MUST sit under
`if __name__ == '__main__'` or each worker re-runs the whole pipeline.
"""
import os
# Telemetry / autolog OFF + don't pip-install on a local box (set before imports).
for k, v in {
    "MLFLOW_AUTOLOGGING_DISABLE": "1", "OTEL_SDK_DISABLED": "true",
    "CREWAI_DISABLE_TELEMETRY": "true", "CREWAI_DO_NOT_TRACK": "1",
    "CREWAI_TELEMETRY_OPT_OUT": "true", "LITELLM_TELEMETRY": "False",
    "POSTHOG_DISABLED": "true", "DIQ_RUNNER_SKIP_INSTALL": "1",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(k, v)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import glob
import pandas as pd

from config.schema import load_config_from_yaml
from utils.deterministic_pipeline import run_full_deterministic_pipeline


def main():
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                        format="%(asctime)s %(levelname).1s %(name)s | %(message)s")
    CAT = "HOME_AND_HYGIENE"
    cfg = load_config_from_yaml("config/config_th_base_local.yaml")
    cfg.input_data_path = f"sourcedata/THAILAND/by_category/{CAT}.parquet"
    cfg.artifact_base_path = f"artifacts_th_local/{CAT}_smoke"
    cfg.output_folder_path = f"artifacts_th_local/{CAT}_smoke"

    # FAST smoke overrides (just to validate the path quickly)
    cfg.design.dmh_n_seeds = 1
    cfg.design.dmh_objective = "tweedie"
    cfg.design.dmh_top_k_features = 40
    cfg.design.parallel_training_workers = 8

    print(f"[smoke] input={cfg.input_data_path}", flush=True)
    print(f"[smoke] key={cfg.prediction_key_cols} ts={cfg.timestamp_col} target={cfg.target_col} "
          f"horizon={cfg.forecast_horizon} run_mode={cfg.run_mode}", flush=True)
    print(f"[smoke] splits train={cfg.train_start}..{cfg.train_end} val={cfg.val_start}..{cfg.val_end} "
          f"test={cfg.test_start}..{cfg.test_end}", flush=True)

    t0 = time.time()
    run_full_deterministic_pipeline(cfg, clean_artifacts=True)
    print(f"\n[smoke] pipeline returned in {time.time()-t0:.0f}s", flush=True)

    cands = glob.glob(f"{cfg.artifact_base_path}/**/inference_forecast.csv", recursive=True) + \
            glob.glob(f"{cfg.artifact_base_path}/**/inference_forecast.parquet", recursive=True)
    print("[smoke] forecast files:", cands, flush=True)
    if cands:
        f = cands[0]
        df = pd.read_csv(f) if f.endswith(".csv") else pd.read_parquet(f)
        print("[smoke] forecast shape:", df.shape, flush=True)
        print("[smoke] columns:", list(df.columns), flush=True)
        print(df.head(15).to_string(), flush=True)
        kc = "key" if "key" in df.columns else cfg.prediction_key_cols[0]
        if kc in df.columns:
            e16 = df[df[kc].astype(str).str.endswith("_E1016")]
            print(f"[smoke] E1016 forecast rows: {len(e16)} across {e16[kc].nunique()} keys", flush=True)
    print("\n[smoke] DONE", flush=True)


if __name__ == "__main__":
    main()
