#!/bin/bash
#SBATCH --job-name=n_pca_analysis
#SBATCH --output=../generated_outputs/pca_analysis_new_all.out
#SBATCH --error=../generated_outputs/pca_analysis_new_all.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Run complete analysis
python3 downstream_tasks/pca_analysis_new.py   --embeddings /scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings \
  --metadata /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv   \
  --aggregation 4weeks   --output-dir ../generated_outputs/temporal_strain_umaps_4weeks_all  