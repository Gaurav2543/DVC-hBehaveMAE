#!/bin/bash
#SBATCH --job-name=normal_embed_3days
#SBATCH --output=normal_embed_3days.out  # Save output logs
#SBATCH --error=normal_embed_3days.err   # Save error logs
#SBATCH --gres=gpu:1  # Number of GPUs requested
#SBATCH --cpus-per-task=8  # Number of CPU threads
#SBATCH --mem=256G  # Memory requested
#SBATCH --time=72:00:00  # Time limit (adjust as needed)
#SBATCH --partition=l40s  # Change this based on your cluster’s GPU partition
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --nodes=1  # Run on a single node

#!/usr/bin/env bash

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python3 downstream_tasks/extract_dvc_embeddings_wrong.py \
    --ckpt_dir /scratch/bhole/dvc_data_normal/models_3days --ckpt_name checkpoint-00050.pth \
    --output_dir /scratch/bhole/dvc_data_normal/extract_dvc_embeddings_3days --skip_missing \
    --dvc_root /scratch/bhole/dvc_data_normal/cage_activations --num_frames 1440 --batch_size 512 \
    --summary_csv /scratch/bhole/dvc_data_normal/summary_files/1440_multithreaded/no_phenotyping_days_greater_than_1440_and_is_trusted_animal_1440.csv
