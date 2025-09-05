#!/bin/bash
#SBATCH --job-name=compare_jsons
#SBATCH --output=logs/compare_jsons.out
#SBATCH --error=logs/compare_jsons.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=h100
#SBATCH --ntasks=1
#SBATCH --nodes=1

BASE_DIR="outputs_predictions_final/shuffled_wo_nb_animals"
OUTPUT_DIR="${BASE_DIR}/comparison_results"

python3 -u downstream_tasks/compare_jsons.py \
  --base_dir "$BASE_DIR" \
  --output_dir "$OUTPUT_DIR" 