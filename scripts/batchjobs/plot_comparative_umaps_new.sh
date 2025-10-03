#!/bin/bash
#SBATCH --job-name=plt_comp_umaps
#SBATCH --output=../generated_outputs/plot_comparative_umaps_all.out
#SBATCH --error=../generated_outputs/plot_comparative_umaps_all.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python downstream_tasks/plot_comparative_umaps_new.py \
  --metadata_path /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv \
  --results_json generated_outputs/umap_analysis/40320_4weeks_7levels/predictions/40320_4weeks_combined.json \
  --output_dir umaps_all_4weeks \
  --embeddings_dir /scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings \
  --aggregation 4weeks \
  --embedding_levels level_1_pooled level_2_pooled level_3_pooled level_4_pooled level_5_pooled level_6_pooled level_7_pooled combined \
  --embedding_labels level1 level2 level3 level4 level5 level6 level7 comb \
  --embedding_lengths '{"level1":"(5min)", "level2":"(30min)", "level3":"(2hrs)", "level4":"(6hrs)", "level5":"(1day)", "level6":"(2days)", "level7":"(4days)", "comb":"(combined)"}' \
  --umap_name_prefix umap_viz_4weeks