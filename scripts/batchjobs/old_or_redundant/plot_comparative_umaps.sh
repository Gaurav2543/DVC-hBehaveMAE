#!/bin/bash
#SBATCH --job-name=plt_comp_umaps
#SBATCH --output=plot_comparative_umaps_all.out
#SBATCH --error=plot_comparative_umaps_all.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

python plot_comparative_umaps.py \
--metadata_path /scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv \
--results_json generated_outputs/umap_analysis/4320_3days_345/predictions/4320_3days_345_comb.json \
--output_dir umaps_all \
--embedding_paths /scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850/test_3days_level1.npy \
    "/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850/test_3days_level2.npy"\
    "/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850/test_3days_level3.npy" \
    "/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850/test_3days_level4.npy" \
    "/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850/test_3days_level5.npy" \
    "/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850/test_3days_comb.npy" \
 --embedding_labels level1 level2 level3 level4 level5 comb \
 --embedding_lengths '{"level1":"(5mins)", "level2":"(30mins)", "level3":"(2hrs)", "level4":"(6hrs)", "level5":"(12hrs)", "comb":"(combined)"}' \
 --umap_name_prefix umap_viz  