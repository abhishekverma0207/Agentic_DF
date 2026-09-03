"""Quick per-key test: does the pipeline's EXISTING per-key path (recursive, all-
individual) beat LEGO on FABRIC_CLEANING? Restricted to E1016 keys for speed
(per-key models use each key's own history). __main__-guarded for spawn safety."""
import os
for k, v in {"MLFLOW_AUTOLOGGING_DISABLE": "1", "OTEL_SDK_DISABLED": "true",
             "CREWAI_DISABLE_TELEMETRY": "true", "CREWAI_DO_NOT_TRACK": "1",
             "DIQ_RUNNER_SKIP_INSTALL": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, time, logging
import numpy as np, pandas as pd
from config.schema import load_config_from_yaml
from utils.deterministic_pipeline import run_full_deterministic_pipeline


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                        format="%(asctime)s %(levelname).1s %(name)s | %(message)s")
    src = "sourcedata/THAILAND/by_category/FABRIC_CLEANING.parquet"
    df = pd.read_parquet(src)
    e16 = df[df["key"].astype(str).str.endswith("_E1016")].copy()
    inp = "sourcedata/THAILAND/by_category/_FABRIC_CLEANING_E1016.parquet"
    e16.to_parquet(inp, index=False)
    print(f"[perkey] FABRIC_CLEANING E1016: {e16['key'].nunique()} keys, {len(e16)} rows", flush=True)

    cfg = load_config_from_yaml("config/config_th_base_local.yaml")
    cfg.input_data_path = inp
    cfg.artifact_base_path = "artifacts_th_local/fc_perkey"
    cfg.output_folder_path = "artifacts_th_local/fc_perkey"
    # PER-KEY (recursive) path, force all keys individual
    cfg.design.use_direct_multi_horizon = False
    mla = cfg.design.model_level_allocation
    mla.max_individual_pct = 1.0
    mla.min_individual_score = 0.0
    mla.min_nonzero_obs_for_individual = 0
    mla.max_zero_fraction_for_individual = 1.0
    mla.volume_override_quantile = 1.0
    mla.forecastability_override = 1.0
    cfg.design.parallel_training_workers = 8

    t0 = time.time()
    run_full_deterministic_pipeline(cfg, clean_artifacts=True)
    print(f"[perkey] pipeline done in {time.time()-t0:.0f}s", flush=True)

    cands = (glob.glob("artifacts_th_local/fc_perkey/**/inference_forecast.csv", recursive=True) +
             glob.glob("artifacts_th_local/fc_perkey/**/inference_forecast.parquet", recursive=True))
    print("[perkey] forecast files:", cands, flush=True)
    if not cands:
        print("[perkey] NO FORECAST (recursive path likely crashed)", flush=True)
        return
    fc = pd.read_csv(cands[0]) if cands[0].endswith(".csv") else pd.read_parquet(cands[0])
    from utils import lego_eval
    base = lego_eval.load_eval_base()
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted")
    p = p[p["Category"] == "FABRIC CLEANING"]

    def ab(s):
        a = s["actual"].values; f = s["ours"].fillna(0).values; t = a.sum()
        return (1 - np.abs(f - a).sum() / t, (f - a).sum() / t) if t > 0 else (np.nan, np.nan)
    print("\n[perkey] FABRIC CLEANING per-key vs LEGO (13wk 0.411/-0.05, t45 0.278/-0.11; global DMH was 0.342/0.163):", flush=True)
    print("  per-key: 13wk %.3f/%+.2f | t45 %.3f/%+.2f" % (*ab(p), *ab(p[p.t.isin([4, 5])])), flush=True)


if __name__ == "__main__":
    main()
