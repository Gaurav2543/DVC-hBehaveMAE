#!/bin/bash
#SBATCH --job-name=3d_BW_emb_pred
#SBATCH --output=../generated_outputs/3d_BW_emb_pred.out
#SBATCH --error=../generated_outputs/3d_BW_emb_pred.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail
export MPLBACKEND=Agg
embedding_aggregation="3days"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# === Define parameters ===
PARENT_ROOT="/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850"
METADATA_PATH=/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv
RESULTS_DIR="../generated_outputs/prediction_BW_embedding_analysis/4320_${embedding_aggregation}"

declare -A EMBEDDINGS=(
  ["level1"]="test_${embedding_aggregation}_level1.npy"
  ["level2"]="test_${embedding_aggregation}_level2.npy"
  ["level3"]="test_${embedding_aggregation}_level3.npy"
  ["level4"]="test_${embedding_aggregation}_level4.npy"
  ["level5"]="test_${embedding_aggregation}_level5.npy"
  ["comb"]="test_${embedding_aggregation}_comb.npy"
)

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

BODYWEIGHT_FILE="/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv"
TISSUE_COLLECTION_FILE="/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv"

# Add after defining the file paths
if [[ ! -f "$BODYWEIGHT_FILE" ]]; then
    echo "ERROR: Bodyweight file not found: $BODYWEIGHT_FILE"
    exit 1
fi

if [[ ! -f "$TISSUE_COLLECTION_FILE" ]]; then
    echo "ERROR: Tissue collection file not found: $TISSUE_COLLECTION_FILE"
    exit 1
fi

echo "✓ Bodyweight files validated"

# === Loop over embedding types ===
for type in "${!EMBEDDINGS[@]}"; do
  EMB_PATH="$PARENT_ROOT/${EMBEDDINGS[$type]}"

  echo "----------------------------------------"
  echo "[START] Exploring raw embeddings for embedding type: $type"
  OUTPUT_DIR="$RESULTS_DIR/explore_raw_embeddings_$type"
  mkdir -p "$OUTPUT_DIR"
  python3 downstream_tasks/explore_raw_embeddings.py --embeddings "$EMB_PATH" --labels "$METADATA_PATH" \
    --output-dir "$OUTPUT_DIR" --strain "A/J" > "$OUTPUT_DIR/explore_raw_embeddings_${type}_A_J.log" 2>&1
  echo "[END] Finished exploring raw embeddings for embedding type: $type"
  echo "Results saved to: $OUTPUT_DIR"
  
  echo "----------------------------------------"
  echo "[START] Evaluating embeddings for regression for embedding type: $type"
  OUTPUT_DIR="$RESULTS_DIR/evaluate_embeddings_regress_$type"
  mkdir -p "$OUTPUT_DIR"
  python3 downstream_tasks/evaluate_embeddings.py --task regression --target-label avg_age_days_chunk_start --group-label strain \
    --embeddings "$EMB_PATH" --labels "$METADATA_PATH" > "$OUTPUT_DIR/evaluate_embeddings_${type}.log" 2>&1
  echo "[END] Finished evaluating embeddings for embedding type: $type"
  echo "Results saved to: $OUTPUT_DIR"

  echo "----------------------------------------"
  echo "[START] Evaluating embeddings for classification for embedding type: $type"
  OUTPUT_DIR="$RESULTS_DIR/evaluate_embeddings_classif_$type"
  mkdir -p "$OUTPUT_DIR"
  python3 downstream_tasks/evaluate_embeddings.py --task classification --target-label strain \
    --embeddings "$EMB_PATH" --labels "$METADATA_PATH" > "$OUTPUT_DIR/evaluate_embeddings_classif_${type}.log" 2>&1
  echo "[END] Finished evaluating embeddings for classification for embedding type: $type"
  echo "Results saved to: $OUTPUT_DIR"

  # echo "----------------------------------------"
  # echo "[START] Running BW integration and analysis for embedding type: $type"
  # OUTPUT_DIR="$RESULTS_DIR/BW_integration_$type"
  # mkdir -p "$OUTPUT_DIR"
  # python3 downstream_tasks/BW_integration_and_analysis.py --embeddings "$EMB_PATH" \
  #   --metadata "$METADATA_PATH" --summary "$METADATA_PATH" --output_dir "$OUTPUT_DIR" \
  #   --bodyweight "$BODYWEIGHT_FILE" --tissue "$TISSUE_COLLECTION_FILE" \
  # > "$OUTPUT_DIR/BW_integration_${type}.log" 2>&1 
  # echo "[END] Finished BW analysis for embedding type: $type"
  # echo "Results saved to: $OUTPUT_DIR"

  echo "----------------------------------------"
  echo "[START] Running analysis for embedding type: $type"
  OUTPUT_DIR="$RESULTS_DIR/run_analysis_$type"
  mkdir -p "$OUTPUT_DIR"
  python run_analysis.py "$EMB_PATH" "$METADATA_PATH"  \
    --bodyweight-file "$BODYWEIGHT_FILE" --tissue-file "$TISSUE_COLLECTION_FILE" \
    --summary-file "$METADATA_PATH" --max-workers 64 --output-dir "$OUTPUT_DIR" \
  > ../generated_outputs/run_analysis_${embedding_aggregation}_"$type".log 2>&1
  echo "[END] Finished analysis for embedding type: $type"
  echo "Results saved to: $OUTPUT_DIR"
  
  echo "----------------------------------------"
  echo "[START] Running temporal subaggregation analysis for embedding type: $type"
  OUTPUT_DIR="$RESULTS_DIR/temporal_subagg_analysis_$type"
  mkdir -p "$OUTPUT_DIR"
  python3 downstream_tasks/temporal_regression_analysis.py "$EMB_PATH" "$METADATA_PATH" \
    --bodyweight_file "$BODYWEIGHT_FILE" --tissue_file "$TISSUE_COLLECTION_FILE" \
    --summary_file "$METADATA_PATH" --output_dir "$OUTPUT_DIR" --max_workers 64 \
    > "$OUTPUT_DIR/temporal_subaggregation_${type}.log" 2>&1
  echo "[END] Finished temporal subaggregation analysis for embedding type: $type"
  echo "Results saved to: $OUTPUT_DIR"
done
