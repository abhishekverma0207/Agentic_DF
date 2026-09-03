#!/bin/bash
# Clean consistent all-9 baseline, sped up: full-mode auto ensemble, all keys, recency=26,
# n_seeds=1 + num_boost_round=1500 (~4x faster, converged quality preserved). Run 2 categories
# concurrently (separate processes -> no cross-cat memory accrual; the single-process all-9
# OOM'd). Order interleaves small+big so two big categories don't overlap. Saves predicted
# (seasonal-corrected) + predicted_raw per category.
cd "/Users/debonilchowdhury/data_science_debonil/Data_Science_Library/IntelligentPlatform/IntelligentPlatform/Agentic_Forecasting" || exit 1
ML=notebooks/rca_outputs/_final_master.log
: > $ML
runcat() {
  CAT=$1
  echo "=== $(date +%H:%M) start $CAT ===" >> "$2"
  python run_th_local.py --mode full --origin final --categories "$CAT" \
      --out-root artifacts_th_local/final > "notebooks/rca_outputs/_final_${CAT}.log" 2>&1
  echo "=== $(date +%H:%M) done $CAT rc=$? ===" >> "$2"
}
export -f runcat
export ML
# interleave small+big so concurrent pair stays memory-balanced
printf '%s\n' BODY FABRIC_CLEANING DEODORANTS_AND_FRAGRANCES HAIR_CARE SKIN_CLEANSING FOODS FABRIC_ENHANCERS FACE HOME_AND_HYGIENE \
  | xargs -P 2 -I {} bash -c 'runcat "$@"' _ {} "$ML"
echo "FINAL_RUN COMPLETE $(date +%H:%M)" >> $ML
