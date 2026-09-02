#!/bin/bash
# Final all-9 rollout: each category in its own process (memory-safe; the single-process
# all-9 run OOM'd). The category GATE auto-routes: strong-global -> global-only (fast),
# broken-global (FABRIC-class) -> blind broad per-key. Then the sales-weighted scorecard.
cd "/Users/debonilchowdhury/data_science_debonil/Data_Science_Library/IntelligentPlatform/IntelligentPlatform/FEUagenticForecastingUpgrade" || exit 1
# FOODS + FABRIC_CLEANING already produced final forecasts via the gate-tests; run the
# other 7 here. The scorecard aggregates all 9 from the saved _hybrid_oos_fc.parquet files.
CATS="BODY DEODORANTS_AND_FRAGRANCES FABRIC_ENHANCERS FACE HAIR_CARE HOME_AND_HYGIENE SKIN_CLEANSING"
for CAT in $CATS; do
  echo "=== $(date +%H:%M) $CAT ===" >> notebooks/rca_outputs/_rollout_master.log
  python scripts/_hybrid_cat.py "$CAT" > "notebooks/rca_outputs/_rollout_${CAT}.log" 2>&1
  rc=$?
  tail -6 "notebooks/rca_outputs/_rollout_${CAT}.log" | grep -iE "decision|HYBRID-OOS|global \(all|WIN|Error" >> notebooks/rca_outputs/_rollout_master.log
  echo "  ($CAT rc=$rc)" >> notebooks/rca_outputs/_rollout_master.log
done
echo "=== $(date +%H:%M) SCORECARD ===" >> notebooks/rca_outputs/_rollout_master.log
python scripts/_rollout_scorecard.py >> notebooks/rca_outputs/_rollout_master.log 2>&1
echo "ROLLOUT COMPLETE" >> notebooks/rca_outputs/_rollout_master.log
