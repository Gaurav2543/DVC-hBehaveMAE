#!/bin/bash
#SBATCH --job-name=pca_analysis
#SBATCH --output=../generated_outputs/pca_analysis.out
#SBATCH --error=../generated_outputs/pca_analysis.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Run complete analysis
python downstream_tasks/pca_analysis.py \
  --embeddings /scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings \
  --metadata /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv \
  --aggregation 4weeks \
  --embedding-level combined \
  --output-dir enhanced_analysis_4weeks_100 \
  --comparison-levels level_2_pooled level_4_pooled level_6_pooled combined

# python3 downstream_tasks/pca_analysis.py --embeddings /scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings   \
#     --metadata /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv   --aggregation 4weeks   \
#     --embedding-level level_6_pooled   --output-dir pca_analysis_4weeks > ../generated_outputs/pca_analysis_4weeks.log 2>&1
