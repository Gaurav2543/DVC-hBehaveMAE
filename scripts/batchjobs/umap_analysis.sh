#!/bin/bash
#SBATCH --job-name=3d_umap_anal_pred
#SBATCH --output=../generated_outputs/3d_umap_anal_pred_345_2.out
#SBATCH --error=../generated_outputs/3d_umap_anal_pred_345_2.err
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
RESULTS_JSON_PREFIX="4320_${embedding_aggregation}_345"
BASE_OUTPUT="../generated_outputs/umap_analysis/${RESULTS_JSON_PREFIX}"
PARENT_ROOT="/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850" # SET HERE THE PARENT ROOT FOR EMBEDDINGS (where the embeddings are stored)
METADATA_PATH="/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" # SET HERE THE METADATA PATH

# Common parameters
EMBEDDING_DIMENSIONS=4320 # Number of dimensions in the embeddings (4320 for 3-day aggregations)  

# Output directories for all results
RESULTS_ROOT="$BASE_OUTPUT/predictions"                 
UMAP_OUTPUT_DIR="$BASE_OUTPUT/umaps"              
CLUSTER_OUTPUT_DIR="$BASE_OUTPUT/global_strain_umaps"      
COMPARISON_OUTPUT_DIR="$BASE_OUTPUT/comparison_results"  
# STRAT_OUTPUT_DIR="$BASE_OUTPUT/strat_nb_animals"  

# UMAP name prefix and embedding lengths (modify for different aggregation types)
UMAP_NAME_PREFIX="umap"
# EMBEDDING_LENGTHS='{"level1":"(5mins)", "level2":"(30min)", "level3":"(2hrs)", "level4":"(6hrs)", "level5":"(12hrs)", "comb":"(concatenated)"}'
# EMBEDDING_LENGTHS='{"level1":"(5mins)", "level2":"(30min)", "level3":"(2hrs)", "level4":"(6hrs)", "level5":"(12hrs)", "level6":"(1day)", "level7":"(2days)", "comb":"(concatenated)"}'
EMBEDDING_LENGTHS='{"level3":"(2hrs)", "level4":"(6hrs)", "level5":"(12hrs)", "comb":"(concatenated)"}'

# Ensure dirs
# mkdir -p "$BASE_OUTPUT" "$RESULTS_ROOT" "$UMAP_OUTPUT_DIR" "$CLUSTER_OUTPUT_DIR" "$COMPARISON_OUTPUT_DIR" "$STRAT_OUTPUT_DIR"
mkdir -p "$BASE_OUTPUT" "$RESULTS_ROOT" "$UMAP_OUTPUT_DIR" "$CLUSTER_OUTPUT_DIR" "$COMPARISON_OUTPUT_DIR"

# Helper: ordered loop to avoid assoc-array randomness
# EMB_ORDER=("level1" "level2" "level3" "level4" "level5" "comb")
EMB_ORDER=("level3" "level4" "level5" "comb")

echo "[INFO] Base output directory: $BASE_OUTPUT"
echo "[INFO] Results root: $RESULTS_ROOT"

declare -A EMBEDDINGS=(
  # ["level1"]="test_${embedding_aggregation}_level1.npy"
  # ["level2"]="test_${embedding_aggregation}_level2.npy"
  ["level3"]="test_${embedding_aggregation}_level3.npy"
  ["level4"]="test_${embedding_aggregation}_level4.npy"
  ["level5"]="test_${embedding_aggregation}_level5.npy"
  # ["level5"]="test_${embedding_aggregation}_level6.npy"
  # ["level5"]="test_${embedding_aggregation}_level7.npy"
  ["comb"]="test_${embedding_aggregation}_comb.npy"
)

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

BODYWEIGHT_FILE="~/phenotypeFormattedSimplified_all_csv_data/BW_data.csv"
TISSUE_COLLECTION_FILE="~/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv"

# ==============================
# STEP 0: Metadata Prediction
# ==============================
# for type in "${EMB_ORDER[@]}"; do
#   echo "----------------------------------------"
#   echo "[STEP 0] Predicting metadata for embedding type: $type"
#   EMB_PATH="$PARENT_ROOT/${EMBEDDINGS[$type]}"
#   OUTPUT_DIR="$BASE_OUTPUT/age_strain_predictor_$type"
#   RESULTS_JSON="${RESULTS_JSON_PREFIX}_${type}.json"
#   mkdir -p "$OUTPUT_DIR"
#   python3 downstream_tasks/age_strain_predictor.py --task both \
#     --sliding-window --window-sizes 15 30 45 60 --window-step 20 --min-samples-per-window 10 \
#     --aggregation-method percentiles --cage-aggregation mean --max-workers 64 \
#     --bodyweight-file "$BODYWEIGHT_FILE" --tissue-collection-file "$TISSUE_COLLECTION_FILE" \
#     --use-covariate --covariate-strategy gam_spline \
#     --target-label avg_age_days_chunk_start --classification-target strain --plot-confusion-matrix \
#     --embeddings "$EMB_PATH" --labels "$METADATA_PATH" --output-dir "$OUTPUT_DIR" --results-json "$RESULTS_JSON" \
#     --summary-metadata "$METADATA_PATH" > ../generated_outputs/age_strain_predictor_${embedding_aggregation}_"$type".log 2>&1
#   echo "[END] Finished age strain prediction for embedding type: $type"
#   echo "Results saved to: $OUTPUT_DIR"
#   echo "Results summary JSON: $RESULTS_JSON"
# done
# echo "[COMPLETE] Step 0 finished."

# ==============================
# STEP 1: UMAP Generation
# ==============================
echo "[STEP 1] Generating UMAPs"

python3 -u downstream_tasks/umap_generation.py \
  --metadata_path "$METADATA_PATH" \
  --results_json "$RESULTS_ROOT" \
  --output_dir "$UMAP_OUTPUT_DIR" \
  --embedding_level3 "$PARENT_ROOT/${EMBEDDINGS[level3]}" \
  --embedding_level4 "$PARENT_ROOT/${EMBEDDINGS[level4]}" \
  --embedding_level5 "$PARENT_ROOT/${EMBEDDINGS[level5]}" \
  --embedding_comb "$PARENT_ROOT/${EMBEDDINGS[comb]}" \
  --embedding_dimensions "$EMBEDDING_DIMENSIONS" \
  --embedding_lengths "$EMBEDDING_LENGTHS" \
  --umap_name_prefix "$UMAP_NAME_PREFIX"
echo "[COMPLETE] Step 1 finished."

  # --embedding_level1 "$PARENT_ROOT/${EMBEDDINGS[level1]}" \
  # --embedding_level2 "$PARENT_ROOT/${EMBEDDINGS[level2]}" \


# ==============================
# STEP 2: UMAP Clustering by Strain
# ==============================
echo "[STEP 2] Running strain-based UMAP clustering"
python3 -u downstream_tasks/umap_clustering_strain.py \
  --metadata_path "$METADATA_PATH" \
  --output_dir "$CLUSTER_OUTPUT_DIR" \
  --embedding_level3 "$PARENT_ROOT/${EMBEDDINGS[level3]}" \
  --embedding_level4 "$PARENT_ROOT/${EMBEDDINGS[level4]}" \
  --embedding_level5 "$PARENT_ROOT/${EMBEDDINGS[level5]}" \
  --embedding_comb "$PARENT_ROOT/${EMBEDDINGS[comb]}" \
  --method mean
echo "[COMPLETE] Step 2 finished."

  # --embedding_level1 "$PARENT_ROOT/${EMBEDDINGS[level1]}" \
  # --embedding_level2 "$PARENT_ROOT/${EMBEDDINGS[level2]}" \

# ==============================
# STEP 3: Compare JSON Results
# ==============================
echo "[STEP 3] Comparing JSON results"
python3 -u downstream_tasks/compare_jsons.py \
  --base_dir "$RESULTS_ROOT" \
  --output_dir "$COMPARISON_OUTPUT_DIR"
echo "[COMPLETE] Step 3 finished."

# # ==============================
# # STEP 4: Stratified Nb. Mice PCA/UMAP
# # ==============================
# echo "[STEP 5] Running PCA/KMeans (stratified by number of animals)"
# MODE="pca_kmeans"          # Options: "pca_kmeans", "umap"
# OUTPUT_PREFIX="pca_kmeans" # Prefix for output files

# # Dummy JSON path (script requires it even if unused)
# DUMMY_RESULTS_JSON="$RESULTS_ROOT/dummy.json"

# python3 -u downstream_tasks/embedding_analysis_nb_animals.py \
#   --metadata_path "$METADATA_PATH" \
#   --results_json "$DUMMY_RESULTS_JSON" \
#   --output_dir "$STRAT_OUTPUT_DIR" \
#   --embedding_level1 "$PARENT_ROOT/${EMBEDDINGS[level1]}" \
#   --embedding_level2 "$PARENT_ROOT/${EMBEDDINGS[level2]}" \
#   --embedding_level3 "$PARENT_ROOT/${EMBEDDINGS[level3]}" \
#   --embedding_level4 "$PARENT_ROOT/${EMBEDDINGS[level4]}" \
#   --embedding_level5 "$PARENT_ROOT/${EMBEDDINGS[level5]}" \
#   --embedding_comb "$PARENT_ROOT/${EMBEDDINGS[comb]}" \
#   --embedding_dimensions "$EMBEDDING_DIMENSIONS" \
#   --embedding_lengths "$EMBEDDING_LENGTHS" \
#   --umap_name_prefix "$OUTPUT_PREFIX" \
#   --mode "$MODE"

# echo "[STEP 4] Running UMAP (strattified by number of animals)"
# MODE="umap"  # Options: "pca_kmeans", "umap"
# OUTPUT_PREFIX="umap"                     

# # Dummy JSON path (script requires it even if unused)
# DUMMY_RESULTS_JSON="$RESULTS_ROOT/dummy.json"

# python3 -u downstream_tasks/embedding_analysis_nb_animals.py \
#   --metadata_path "$METADATA_PATH" \
#   --results_json "$DUMMY_RESULTS_JSON" \
#   --output_dir "$STRAT_OUTPUT_DIR" \
#   --embedding_level1 "$PARENT_ROOT/${EMBEDDINGS[level1]}" \
#   --embedding_level2 "$PARENT_ROOT/${EMBEDDINGS[level2]}" \
#   --embedding_level3 "$PARENT_ROOT/${EMBEDDINGS[level3]}" \
#   --embedding_level4 "$PARENT_ROOT/${EMBEDDINGS[level4]}" \
#   --embedding_level5 "$PARENT_ROOT/${EMBEDDINGS[level5]}" \
#   --embedding_comb "$PARENT_ROOT/${EMBEDDINGS[comb]}" \
#   --embedding_dimensions "$EMBEDDING_DIMENSIONS" \
#   --embedding_lengths "$EMBEDDING_LENGTHS" \
#   --umap_name_prefix "$OUTPUT_PREFIX" \
#   --mode "$MODE"

# echo "[COMPLETE] Step 4 finished."

echo "Full downstream pipeline completed. Outputs in: $BASE_OUTPUT"



