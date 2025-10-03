#!/bin/bash
#SBATCH --job-name=msc_40320_4w
#SBATCH --output=../generated_outputs/msc_40320_4w.out
#SBATCH --error=../generated_outputs/msc_40320_4w.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Define common parameters
EMBEDDINGS_DIR="/scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings"
METADATA_PATH="/scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv"
SUMMARY_FILE="/scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv"
BODYWEIGHT_FILE="/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv"
TISSUE_FILE="/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv"
AGGREGATION="4weeks"

# Array of embedding levels to process
declare -a LEVELS=("level_1_pooled" "level_2_pooled" "level_3_pooled" "level_4_pooled" "level_5_pooled" "level_6_pooled" "level_7_pooled" "combined")
declare -a LABELS=("l1" "l2" "l3" "l4" "l5" "l6" "l7" "comb")

# Process each embedding level
for i in "${!LEVELS[@]}"; do
    LEVEL="${LEVELS[$i]}"
    LABEL="${LABELS[$i]}"
    
    echo "Processing ${LABEL} (${LEVEL})..."
    
    python3 downstream_tasks/misclassified_strains_new.py \
        "$EMBEDDINGS_DIR" \
        "$METADATA_PATH" \
        --summary_file "$SUMMARY_FILE" \
        --bodyweight_file "$BODYWEIGHT_FILE" \
        --tissue_file "$TISSUE_FILE" \
        --output_dir "../generated_outputs/misclassified_strains_40320_4w_${LABEL}" \
        --max_workers 64 \
        > "misclassified_strains_40320_4w_${LABEL}.log" 2>&1
    
    echo "Completed ${LABEL}"
done

echo "All embedding levels processed"