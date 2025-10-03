#!/bin/bash
#SBATCH --job-name=4320_3d_age_strain
#SBATCH --output=4320_3d_age_strain.out
#SBATCH --error=4320_3d_age_strain.err
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --nodes=1

# === Define parameters ===
METADATA_PATH=/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv
PARENT_ROOT="/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850"
RESULTS_DIR="age_strain_predictions/4320_1day"
RESULTS_JSON_PREFIX="4320_1day"

declare -A EMBEDDINGS=(
  ["level1"]="test_1day_level1.npy"
  ["level2"]="test_1day_level2.npy"
  ["level3"]="test_1day_level3.npy"
  ["level4"]="test_1day_level4.npy"
  ["level4"]="test_1day_level5.npy"
  ["comb"]="test_1day_comb.npy"
)

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

# === Loop over embedding types ===
for type in "${!EMBEDDINGS[@]}"; do
  echo "----------------------------------------"
  echo "[START] Running prediction for embedding type: $type"
  EMB_PATH="$PARENT_ROOT/${EMBEDDINGS[$type]}"
  OUTPUT_DIR="$RESULTS_DIR/3days_$type"
  RESULTS_JSON="${RESULTS_JSON_PREFIX}_${type}.json"

  python3 downstream_tasks/age_strain_predictor.py --task both \
  --sliding-window --window-sizes 30 45 60 --window-step 15 --min-samples-per-window 10 \
  --aggregation-method percentiles --cage-aggregation mean --max-workers 8 \
  --bodyweight-file ~/phenotypeFormattedSimplified_all_csv_data/BW_data.csv \
  --use-covariate --covariate-strategy gam_spline \
  --target-label avg_age_days_chunk_start --classification-target strain --plot-confusion-matrix \
  --tissue-collection-file ~/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv \
  --embeddings "$EMB_PATH" \
  --labels "$METADATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --results-json "$RESULTS_JSON" \
  --summary-metadata "$METADATA_PATH" 

  echo "[END] Finished prediction for embedding type: $type"
  echo "Results saved to: $OUTPUT_DIR"
  echo "Results JSON: $RESULTS_JSON"
  echo "----------------------------------------"
done
