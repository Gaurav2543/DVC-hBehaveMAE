#!/bin/bash
#SBATCH --job-name=msc_4320_3d
#SBATCH --output=../generated_outputs/msc_4320_3d.out
#SBATCH --error=../generated_outputs/msc_4320_3d.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python3 downstream_tasks/misclassified_strains.py \
    "/scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_fixed/test_4weeks_comb.npy" \
    "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --summary_file "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --bodyweight_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv" \
    --tissue_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv" \
    --output_dir ../generated_outputs/misclassified_strains_4320_3d_comb --max_workers 64 > misclassified_strains_4320_3d_comb.log 2>&1

python3 downstream_tasks/misclassified_strains.py \
    "/scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_fixed/test_4weeks_level1.npy" \
    "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --summary_file "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --bodyweight_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv" \
    --tissue_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv" \
    --output_dir ../generated_outputs/misclassified_strains_4320_3d_l1 --max_workers 64 > misclassified_strains_4320_3d_l1.log 2>&1

python3 downstream_tasks/misclassified_strains.py \
    "/scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_fixed/test_4weeks_level2.npy" \
    "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --summary_file "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --bodyweight_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv" \
    --tissue_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv" \
    --output_dir ../generated_outputs/misclassified_strains_4320_3d_l2 --max_workers 64 > misclassified_strains_4320_3d_l2.log 2>&1

python3 downstream_tasks/misclassified_strains.py \
    "/scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_fixed/test_4weeks_level3.npy" \
    "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --summary_file "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --bodyweight_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv" \
    --tissue_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv" \
    --output_dir ../generated_outputs/misclassified_strains_4320_3d_l3 --max_workers 64 > misclassified_strains_4320_3d_l3.log 2>&1

python3 downstream_tasks/misclassified_strains.py \
    "/scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_fixed/test_4weeks_level4.npy" \
    "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --summary_file "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --bodyweight_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv" \
    --tissue_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv" \
    --output_dir ../generated_outputs/misclassified_strains_4320_3d_l4 --max_workers 64 > misclassified_strains_4320_3d_l4.log 2>&1

python3 downstream_tasks/misclassified_strains.py \
    "/scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_fixed/test_4weeks_level5.npy" \
    "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --summary_file "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --bodyweight_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv" \
    --tissue_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv" \
    --output_dir ../generated_outputs/misclassified_strains_4320_3d_l5 --max_workers 64 > misclassified_strains_4320_3d_l5.log 2>&1

python3 downstream_tasks/misclassified_strains.py \
    "/scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_fixed/test_4weeks_level6.npy" \
    "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --summary_file "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --bodyweight_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv" \
    --tissue_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv" \
    --output_dir ../generated_outputs/misclassified_strains_4320_3d_l6 --max_workers 64 > misclassified_strains_4320_3d_l6.log 2>&1

python3 downstream_tasks/misclassified_strains.py \
    "/scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_fixed/test_4weeks_level7.npy" \
    "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --summary_file "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv" \
    --bodyweight_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv" \
    --tissue_file "/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv" \
    --output_dir ../generated_outputs/misclassified_strains_4320_3d_l7 --max_workers 64 > misclassified_strains_4320_3d_l7.log 2>&1

