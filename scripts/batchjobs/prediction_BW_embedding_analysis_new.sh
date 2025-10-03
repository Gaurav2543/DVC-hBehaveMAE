#!/bin/bash
#SBATCH --job-name=4w_BW_emb_pred
#SBATCH --output=../generated_outputs/4w_BW_emb_pred.out
#SBATCH --error=../generated_outputs/4w_BW_emb_pred.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail
export MPLBACKEND=Agg
embedding_aggregation="4weeks"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# === Define parameters ===
EMBEDDINGS_DIR="/scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings"
METADATA_PATH=/scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv
RESULTS_DIR="../generated_outputs/prediction_BW_embedding_analysis/40320_${embedding_aggregation}"

# NEW: Define embedding levels (7 levels + combined)
declare -a EMBEDDING_LEVELS=("level_1_pooled" "level_2_pooled" "level_3_pooled" "level_4_pooled" "level_5_pooled" "level_6_pooled" "level_7_pooled" "combined")

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

BODYWEIGHT_FILE="/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv"
TISSUE_COLLECTION_FILE="/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv"

# Validate bodyweight files
if [[ ! -f "$BODYWEIGHT_FILE" ]]; then
    echo "ERROR: Bodyweight file not found: $BODYWEIGHT_FILE"
    exit 1
fi

if [[ ! -f "$TISSUE_COLLECTION_FILE" ]]; then
    echo "ERROR: Tissue collection file not found: $TISSUE_COLLECTION_FILE"
    exit 1
fi

echo "✓ Bodyweight files validated"

# === Loop over embedding levels ===
for level in "${EMBEDDING_LEVELS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "PROCESSING EMBEDDING LEVEL: $level"
  echo "========================================================================"
  
  echo "----------------------------------------"
  echo "[START] Exploring raw embeddings for level: $level"
  OUTPUT_DIR="$RESULTS_DIR/explore_raw_embeddings_$level"
  mkdir -p "$OUTPUT_DIR"
  python3 downstream_tasks/explore_raw_embeddings_new.py \
    --embeddings "$EMBEDDINGS_DIR" \
    --labels "$METADATA_PATH" \
    --aggregation "$embedding_aggregation" \
    --embedding-level "$level" \
    --strain "A/J" \
    --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/explore_raw_embeddings_${level}_A_J.log" 2>&1
  echo "[END] Finished exploring raw embeddings for level: $level"
  echo "Results saved to: $OUTPUT_DIR"
  
  echo "----------------------------------------"
  echo "[START] Evaluating embeddings for regression for level: $level"
  OUTPUT_DIR="$RESULTS_DIR/evaluate_embeddings_regress_$level"
  mkdir -p "$OUTPUT_DIR"
  python3 downstream_tasks/evaluate_embeddings_new.py \
    --task regression \
    --target-label avg_age_days_chunk_start \
    --group-label strain \
    --embeddings "$EMBEDDINGS_DIR" \
    --labels "$METADATA_PATH" \
    --aggregation "$embedding_aggregation" \
    --embedding-level "$level" \
    > "$OUTPUT_DIR/evaluate_embeddings_${level}.log" 2>&1
  echo "[END] Finished evaluating embeddings for level: $level"
  echo "Results saved to: $OUTPUT_DIR"

  echo "----------------------------------------"
  echo "[START] Evaluating embeddings for classification for level: $level"
  OUTPUT_DIR="$RESULTS_DIR/evaluate_embeddings_classif_$level"
  mkdir -p "$OUTPUT_DIR"
  python3 downstream_tasks/evaluate_embeddings_new.py \
    --task classification \
    --target-label strain \
    --embeddings "$EMBEDDINGS_DIR" \
    --labels "$METADATA_PATH" \
    --aggregation "$embedding_aggregation" \
    --embedding-level "$level" \
    > "$OUTPUT_DIR/evaluate_embeddings_classif_${level}.log" 2>&1
  echo "[END] Finished evaluating embeddings for classification for level: $level"
  echo "Results saved to: $OUTPUT_DIR"

  echo "----------------------------------------"
  echo "[START] Running BW integration and analysis for level: $level"
  OUTPUT_DIR="$RESULTS_DIR/BW_integration_$level"
  mkdir -p "$OUTPUT_DIR"
  python3 downstream_tasks/BW_integration_and_analysis_new.py \
    --embeddings "$EMBEDDINGS_DIR" \
    --metadata "$METADATA_PATH" \
    --summary "$METADATA_PATH" \
    --bodyweight "$BODYWEIGHT_FILE" \
    --tissue "$TISSUE_COLLECTION_FILE" \
    --aggregation "$embedding_aggregation" \
    --embedding-level "$level" \
    --output_dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/BW_integration_${level}.log" 2>&1 
  echo "[END] Finished BW analysis for level: $level"
  echo "Results saved to: $OUTPUT_DIR"

  echo "----------------------------------------"
  echo "[START] Running comprehensive analysis for level: $level"
  OUTPUT_DIR="$RESULTS_DIR/run_analysis_$level"
  mkdir -p "$OUTPUT_DIR"
  python run_analysis_new.py "$EMBEDDINGS_DIR" "$METADATA_PATH" \
    --bodyweight-file "$BODYWEIGHT_FILE" \
    --tissue-file "$TISSUE_COLLECTION_FILE" \
    --summary-file "$METADATA_PATH" \
    --aggregation "$embedding_aggregation" \
    --embedding-level "$level" \
    --max-workers 64 \
    --output-dir "$OUTPUT_DIR" \
    > ../generated_outputs/run_analysis_${embedding_aggregation}_"$level".log 2>&1
  echo "[END] Finished comprehensive analysis for level: $level"
  echo "Results saved to: $OUTPUT_DIR"
  
  echo "----------------------------------------"
  echo "[START] Running temporal subaggregation analysis for level: $level"
  OUTPUT_DIR="$RESULTS_DIR/temporal_subagg_analysis_$level"
  mkdir -p "$OUTPUT_DIR"
  python3 downstream_tasks/temporal_regression_analysis_new.py \
    "$EMBEDDINGS_DIR" "$METADATA_PATH" \
    --bodyweight_file "$BODYWEIGHT_FILE" \
    --tissue_file "$TISSUE_COLLECTION_FILE" \
    --summary_file "$METADATA_PATH" \
    --aggregation "$embedding_aggregation" \
    --embedding-level "$level" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers 64 \
    > "$OUTPUT_DIR/temporal_subaggregation_${level}.log" 2>&1
  echo "[END] Finished temporal subaggregation analysis for level: $level"
  echo "Results saved to: $OUTPUT_DIR"
  
  echo ""
  echo "✓ COMPLETED ALL ANALYSES FOR LEVEL: $level"
  echo "========================================================================"
done

echo ""
echo "========================================================================"
echo "ALL ANALYSES COMPLETED FOR ALL 8 EMBEDDING LEVELS!"
echo "========================================================================"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Summary of processed levels:"
for level in "${EMBEDDING_LEVELS[@]}"; do
  echo "  - $level"
done
echo ""
echo "Check individual log files in each output directory for details."