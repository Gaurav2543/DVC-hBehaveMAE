#!/bin/bash
#SBATCH --job-name=HBehaveMAE_train
#SBATCH --output=logs/train_hBehaveMAE_%j.out  # Save output logs
#SBATCH --error=logs/train_hBehaveMAE_%j.err   # Save error logs
#SBATCH --gres=gpu:1  # Number of GPUs requested
#SBATCH --cpus-per-task=8  # Number of CPU threads
#SBATCH --mem=32G  # Memory requested
#SBATCH --time=12:00:00  # Time limit (adjust as needed)
#SBATCH --partition=l40s  # Change this based on your cluster’s GPU partition
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --nodes=1  # Run on a single node

# Load modules (if necessary)
module load gcc/13.2.0
module load cuda/12.4.1
source ~/miniconda3/bin/activate # If using Anaconda

# Activate your environment
conda activate behavemae  # Ensure this is the correct Conda environment

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Define number of GPUs
GPUS=1  # Adjust based on `--gres=gpu:X`

# Run the script
bash scripts/dvc/train_hBehaveMAE.sh $GPUS
