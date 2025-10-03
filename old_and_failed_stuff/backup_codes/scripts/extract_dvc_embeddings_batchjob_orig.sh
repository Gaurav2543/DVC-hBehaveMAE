#!/bin/bash
#SBATCH --job-name=embed_hBehaveMAE
#SBATCH --output=extract_embed_gen_hiera2.out  # Save output logs
#SBATCH --error=extract_embed_gen_hiera2.err   # Save error logs
#SBATCH --gres=gpu:1  # Number of GPUs requested
#SBATCH --cpus-per-task=8  # Number of CPU threads
#SBATCH --mem=32G  # Memory requested
#SBATCH --time=72:00:00  # Time limit (adjust as needed)
#SBATCH --partition=l40s  # Change this based on your cluster’s GPU partition
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --nodes=1  # Run on a single node

#!/usr/bin/env bash

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PRETRAINED_MODEL_DIR="output2" # Where your checkpoint-best.pth is
EMBEDDINGS_SAVE_TO="extracted_embeddings_z"
DVC_DATA_DIR="/scratch/bhole/data" # Contains summary table and CSVs
# STATS_FILE_PATH="${DVC_DATA_DIR}/dvc_zscore_stats.npy" # If using z-score

python "downstream_tasks/extract_dvc_embeddings.py" \
    --dvc_data_root_dir "${DVC_DATA_DIR}" \
    --dvc_summary_table_filename "summary_table_imputed_with_sets_sub_20_CompleteAge_Strains.csv" \
    --model_checkpoint_dir "${PRETRAINED_MODEL_DIR}" \
    --model_checkpoint_filename "checkpoint-00099.pth" \
    --embeddings_output_dir "${EMBEDDINGS_SAVE_TO}" \
    --normalization_method "percentage" \
    --model_name "gen_hiera" \
    --input_size 1440 1 12 \
    --patch_kernel 2 1 12 \
    --stages 2 3 4 \
    --q_strides "15,1,1;6,1,1" \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 64 96 128 \
    --mask_unit_attn True False False \
    --sep_pos_embed True \
    --num_frames_per_window 1440 \
    --sliding_window_step 1 \
    --combine_stages True \
    --pca_target_dim 64 \
    --batch_size_inf 32 \
    --fp16 True \
    --skip_missing_electrodes
