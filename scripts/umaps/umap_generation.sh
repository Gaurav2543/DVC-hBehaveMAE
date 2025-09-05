#!/bin/bash
#SBATCH --job-name=umaps_generation
#SBATCH --output=logs/umap_generation_final_s.out
#SBATCH --error=logs/umap_generation_final_s.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=h100
#SBATCH --ntasks=1
#SBATCH --nodes=1

# Set parameters
METADATA_PATH="/scratch/bhole/dvc_data_old/smoothed/smoothed_dvc_shuffled_data_1_1/dvc_data_1/summary_shuffled_A.csv"
#METADATA_PATH="/scratch/bhole/dvc_data_old/smoothed/smoothed_dvc_shuffled_data_1_wo_nb_animals/dvc_data_1/summary_shuffled_A.csv"
#METADATA_PATH=/scratch/lecumber/dvc_data/dvc_data_jon/final_summary_metadata_1440.csv

RESULTS_JSON="outputs_predictions_final/shuffled/dummy.json"
OUTPUT_DIR="umap_plots_final/shuffled/1day"

PARENT_ROOT="/scratch/bhole/dvc_data_old/smoothed/smoothed_dvc_shuffled_data_1_1/dvc_data_1/models_3days/embeddings"
#PARENT_ROOT="/scratch/bhole/dvc_data_old/smoothed/smoothed_dvc_shuffled_data_1_wo_nb_animals/dvc_data_1/models_3days/embeddings"
#PARENT_ROOT="/scratch/lecumber/embeddings_bhole"

EMBEDDING_LOW="$PARENT_ROOT/test_3days_low.npy"
EMBEDDING_MID="$PARENT_ROOT/test_3days_mid.npy"
EMBEDDING_HIGH="$PARENT_ROOT/test_3days_high.npy"
EMBEDDING_COMB="$PARENT_ROOT/test_3days_comb.npy"

EMBEDDING_DIMENSIONS=1440 # LENGHT IN MINUTES

EMBEDDING_LENGTHS='{"low":"(2mins)", "mid":"(30mins)", "high":"(3hs)", "comb":"(concatenated)"}'
OUTPUT_PREFIX="umap"


# Run Python script with arguments
python3 -u downstream_tasks/umap_generation.py \
  --metadata_path "$METADATA_PATH" \
  --results_json "$RESULTS_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --embedding_low "$EMBEDDING_LOW" \
  --embedding_mid "$EMBEDDING_MID" \
  --embedding_high "$EMBEDDING_HIGH" \
  --embedding_comb "$EMBEDDING_COMB" \
  --embedding_dimensions "$EMBEDDING_DIMENSIONS" \
  --embedding_lengths "$EMBEDDING_LENGTHS" \
  --umap_name_prefix "$OUTPUT_PREFIX"

