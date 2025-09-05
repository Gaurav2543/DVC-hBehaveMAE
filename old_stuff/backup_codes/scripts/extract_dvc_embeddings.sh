#!/bin/bash
#SBATCH --job-name=embed2_new_hBehaveMAE
#SBATCH --output=extract_embed2_hbehavemae2_new2.out  # Save output logs
#SBATCH --error=extract_embed2_hbehavemae2_new2.err   # Save error logs
#SBATCH --gres=gpu:1  # Number of GPUs requested
#SBATCH --cpus-per-task=8  # Number of CPU threads
#SBATCH --mem=24G  # Memory requested
#SBATCH --time=72:00:00  # Time limit (adjust as needed)
#SBATCH --partition=l40s  # Change this based on your cluster’s GPU partition
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --nodes=1  # Run on a single node

#!/usr/bin/env bash

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python3 downstream_tasks/extract_dvc_embeddings.py \
    --dvc_root /scratch/bhole/data \
    --ckpt_dir model_checkpoints \
    --ckpt_name checkpoint-00099.pth \
    --output_dir extracted_embeddings \
    --skip_missing
