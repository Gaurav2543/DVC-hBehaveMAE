#!/bin/bash
#SBATCH --job-name=1d_smoothed
#SBATCH --output=logs/train_hBehaveMAE_%j.out  # Save output logs
#SBATCH --error=logs/train_hBehaveMAE_%j.err   # Save error logs
#SBATCH --gres=gpu:1  # Number of GPUs requested
#SBATCH --cpus-per-task=8  # Number of CPU threads
#SBATCH --mem=128G  # Memory requested
#SBATCH --time=72:00:00  # Time limit (adjust as needed)
#SBATCH --partition=h100  # Change this based on your cluster’s GPU partition
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --nodes=1  # Run on a single node

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29502

GPUS=1  

bash scripts/train_test/train_dvc_hBehaveMAE_smoothed.sh $GPUS