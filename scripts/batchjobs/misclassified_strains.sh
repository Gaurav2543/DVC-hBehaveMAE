#!/bin/bash
#SBATCH --job-name=msc_3d_l5
#SBATCH --output=../generated_outputs/msc_4320_3d_l5.out
#SBATCH --error=../generated_outputs/msc_4320_3d_l5.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python3 downstream_tasks/misclassified_strains.py \
    "/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850/test_3days_level5.npy" \
    "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --summary_file "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --bodyweight_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv" \
    --tissue_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv" \
    --output_dir ../generated_outputs/misclassified_strains_4320_3d_l5 --max_workers 64 

