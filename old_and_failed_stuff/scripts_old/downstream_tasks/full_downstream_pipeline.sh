#!/bin/bash
#SBATCH --job-name=3d_analysis
#SBATCH --output=downstream_analysis_3days/pipeline.out
#SBATCH --error=downstream_analysis_3days/pipeline.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail
export MPLBACKEND=Agg
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# ==============================
# CONFIGURATION
# ==============================

# Base output directory
BASE_OUTPUT="downstream_analysis_3days"

# Input paths
METADATA_PATH="/scratch/bhole/dvc_data/smoothed/4320/final_summary_metadata_4320.csv"
EMBEDDINGS_DIR="/scratch/bhole/dvc_data/smoothed/models_3days/embeddings"

# Analysis parameters
AGGREGATION_METHODS="mean median percentiles"  # Space-separated list
TEST_SIZE=0.2
RANDOM_STATE=42

# Sliding window parameters
WINDOW_SIZE=30      # Age window size in days
WINDOW_STEP=15      # Age window step size in days
MIN_SAMPLES=10      # Minimum samples per window

# Analysis toggles (set to true to skip)
SKIP_UMAPS=false
SKIP_CLUSTERING=false

# ==============================
# SETUP
# ==============================

echo "[INFO] Starting unified downstream analysis pipeline"
echo "[INFO] Base output directory: $BASE_OUTPUT"
echo "[INFO] Metadata path: $METADATA_PATH"
echo "[INFO] Embeddings directory: $EMBEDDINGS_DIR"

# Create output directory
mkdir -p "$BASE_OUTPUT"

# Check if inputs exist
if [ ! -f "$METADATA_PATH" ]; then
    echo "[ERROR] Metadata file not found: $METADATA_PATH"
    exit 1
fi

if [ ! -d "$EMBEDDINGS_DIR" ]; then
    echo "[ERROR] Embeddings directory not found: $EMBEDDINGS_DIR"
    exit 1
fi

# ==============================
# RUN PIPELINE
# ==============================

echo "[START] Running unified downstream analysis..."

# Build arguments
ARGS="--metadata_path $METADATA_PATH"
ARGS="$ARGS --embeddings_dir $EMBEDDINGS_DIR"
ARGS="$ARGS --output_dir $BASE_OUTPUT"
ARGS="$ARGS --aggregation_methods $AGGREGATION_METHODS"
ARGS="$ARGS --test_size $TEST_SIZE"
ARGS="$ARGS --random_state $RANDOM_STATE"
ARGS="$ARGS --window_size $WINDOW_SIZE"
ARGS="$ARGS --window_step $WINDOW_STEP"
ARGS="$ARGS --min_samples_per_window $MIN_SAMPLES"

# Add optional flags
if [ "$SKIP_UMAPS" = true ]; then
    ARGS="$ARGS --skip_umaps"
fi

if [ "$SKIP_CLUSTERING" = true ]; then
    ARGS="$ARGS --skip_clustering"
fi

# Run the unified pipeline
python3 -u downstream_tasks/downstream_pipeline.py $ARGS

# ==============================
# COMPLETION
# ==============================

echo "[COMPLETE] Unified downstream analysis finished!"
echo "Results saved to: $BASE_OUTPUT"

# Print summary of outputs
echo ""
echo "=== OUTPUT SUMMARY ==="
echo "Main results directory: $BASE_OUTPUT"
echo ""

if [ -d "$BASE_OUTPUT/sliding_window_plots" ]; then
    echo "Sliding window trend plots:"
    ls -la "$BASE_OUTPUT/sliding_window_plots/"*.pdf 2>/dev/null || echo "  (No PDF files found)"
fi

if [ -d "$BASE_OUTPUT/comparison" ]; then
    echo ""
    echo "Comparison results:"
    ls -la "$BASE_OUTPUT/comparison/"
fi

if [ -d "$BASE_OUTPUT/sliding_windows" ]; then
    echo ""
    echo "Individual sliding window results:"
    echo "  Number of configurations: $(ls -1 "$BASE_OUTPUT/sliding_windows/" | wc -l)"
fi

if [ -d "$BASE_OUTPUT/umaps" ] && [ "$SKIP_UMAPS" = false ]; then
    echo ""
    echo "UMAP visualizations:"
    ls -la "$BASE_OUTPUT/umaps/"*.pdf 2>/dev/null || echo "  (No UMAP files found)"
fi

echo ""
echo "Pipeline metadata: $BASE_OUTPUT/pipeline_metadata.json"
echo ""
echo "=== PIPELINE COMPLETED SUCCESSFULLY ==="