#!/bin/bash
#SBATCH --job-name=720_12h_age_strain
#SBATCH --output=720_12h_age_strain.out
#SBATCH --error=720_12h_age_strain.err
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --nodes=1

# === Define parameters ===
METADATA_PATH=/scratch/bhole/dvc_data/smoothed/720/final_summary_metadata_720.csv
PARENT_ROOT="/scratch/bhole/dvc_data/smoothed/models_12hrs/embeddings"
RESULTS_DIR="age_strain_predictions/720_12hrs"
RESULTS_JSON_PREFIX="720_12hrs"

declare -A EMBEDDINGS=(
  ["low"]="test_12hr_level1.npy"
  ["mid"]="test_12hr_level2.npy"
  ["high"]="test_12hr_level3.npy"
  ["comb"]="test_12hr_comb.npy"
)

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

# === Loop over embedding types ===
for type in "${!EMBEDDINGS[@]}"; do
  echo "[START] Running prediction for embedding type: $type"
  EMB_PATH="$PARENT_ROOT/${EMBEDDINGS[$type]}"
  OUTPUT_DIR="$RESULTS_DIR/12hrs_$type"
  RESULTS_JSON="${RESULTS_JSON_PREFIX}_${type}.json"

  python3 -u downstream_tasks/predict_age_strain_og.py --task both \
    --labels "$METADATA_PATH" --target-label avg_age_days_chunk_start \
    --classification-target strain --aggregation-method percentiles \
    --plot-confusion-matrix \
    --embeddings "$EMB_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --results-json "$RESULTS_JSON" 
done

