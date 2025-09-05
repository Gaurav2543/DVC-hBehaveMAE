#!/bin/bash
#SBATCH --job-name=3days_normal_temp
#SBATCH --output=3days_normal_temp.out  # Save output logs
#SBATCH --error=3days_normal_temp.err   # Save error logs
#SBATCH --gres=gpu:1  # Number of GPUs requested
#SBATCH --cpus-per-task=8  # Number of CPU threads
#SBATCH --mem=256G  # Memory requested
#SBATCH --time=72:00:00  # Time limit (adjust as needed)
#SBATCH --partition=l40s  # Change this based on your cluster’s GPU partition
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --nodes=1  # Run on a single node

#!/usr/bin/env bash

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python temp.py --dataset dvc \
    --path_to_data_dir /scratch/bhole/dvc_data_normal/cage_activations \
    --batch_size 2048 \
    --model hbehavemae \
    --input_size 1440 1 12 \
    --stages 2 3 4 \
    --mask_unit_attn True False False \
    --patch_kernel 2 1 12 \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 64 96 128 \
    --num_frames 1440 \
    --pin_mem \
    --num_workers 8 \
    --output_dir /scratch/bhole/dvc_data_normal/models_3days \
    --summary_csv /scratch/bhole/dvc_data_normal/summary_files/1440_multithreaded/no_phenotyping_days_greater_than_1440_and_is_trusted_animal_1440.csv
