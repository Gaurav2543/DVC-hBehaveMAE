#!/bin/bash
#SBATCH --job-name=t2_4320_1d_BW
#SBATCH --output=t2_4320_1d_BW.out
#SBATCH --error=t2_4320_1d_BW.err
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --nodes=1

# === Define parameters ===
METADATA_PATH=/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv
PARENT_ROOT="/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850"
RESULTS_DIR="age_strain_predictions_new/4320_1day_test2"

declare -A EMBEDDINGS=(
  ["level4"]="test_1day_level4.npy"
)

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

# === Loop over embedding types ===
for type in "${!EMBEDDINGS[@]}"; do
  echo "----------------------------------------"
  echo "[START] Running prediction for embedding type: $type"
  EMB_PATH="$PARENT_ROOT/${EMBEDDINGS[$type]}"
  OUTPUT_DIR="$RESULTS_DIR/1day_$type"

  python run_analysis.py "$EMB_PATH" "$METADATA_PATH"  \
  --bodyweight-file ~/phenotypeFormattedSimplified_all_csv_data/BW_data.csv \
  --tissue-file ~/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv \
  --summary-file "$METADATA_PATH" \
  --max-workers 64 \
  --output-dir "$OUTPUT_DIR" 
  echo "[END] Finished prediction for embedding type: $type"
  echo "Results saved to: $OUTPUT_DIR"
  echo "----------------------------------------"
done

