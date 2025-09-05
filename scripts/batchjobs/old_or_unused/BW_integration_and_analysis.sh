#!/bin/bash
#SBATCH --job-name=BW_4320_1d_l1_anal
#SBATCH --output=BW_4320_1d_l1_anal.out
#SBATCH --error=BW_4320_1d_l1_anal.err
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --nodes=1

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python3 downstream_tasks/BW_integration_and_analysis.py