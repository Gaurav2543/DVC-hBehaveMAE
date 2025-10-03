#!/bin/bash
#SBATCH --job-name=l3_extract_opt
#SBATCH --output=../generated_outputs/extract_opt_long3.out  # Save output logs
#SBATCH --error=../generated_outputs/extract_opt_long3.err   # Save error logs
#SBATCH --gres=gpu:1  # Number of GPUs requested
#SBATCH --cpus-per-task=64  # Number of CPU threads
#SBATCH --mem=370G  # Memory requested
#SBATCH --time=72:00:00  # Time limit (adjust as needed)
#SBATCH --partition=h100  # Change this based on your cluster’s GPU partition
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --nodes=1  # Run on a single node

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "Extracting embeddings..."
python3 downstream_tasks/extract_embeddings_optimized3.py \
  --dvc_root /scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final \
  --summary_csv /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv \
  --ckpt_dir /scratch/bhole/dvc_data/smoothed/models_4weeks \
  --ckpt_name checkpoint-09999.pth \
  --zarr_path /scratch/bhole/dvc_data/smoothed/models_4weeks/extract_opt_long3_zarr \
  --output_dir /scratch/bhole/dvc_data/smoothed/models_4weeks/extract_opt_long3 \
  --time_aggregations 40320 \
  --aggregation_names 4weeks \
  --batch_size 192 --include_spatial
echo "Embedding extraction completed"



# export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# echo "Extracting embeddings..."
# python3 extract_opt.py \
#   --dvc_root /scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final \
#   --summary_csv /scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv \
#   --ckpt_path /scratch/bhole/dvc_data/smoothed/models_3days/checkpoint-09999.pth \
#   --zarr_path /scratch/bhole/dvc_data/smoothed/models_3days/extract_opt_3_zarr \
#   --output_dir /scratch/bhole/dvc_data/smoothed/models_3days/extract_opt_3 \
#   --time_aggregations 4320 \
#   --aggregation_names 3days \
#   --batch_size 32 \
#   --model hbehavemae \
#   --stages 2 3 4 4 5 \
#   --q_strides "6,1,1;4,1,1;3,1,1;2,1,1" \
#   --mask_unit_attn True False False False False \
#   --patch_kernel 5 1 12 \
#   --init_embed_dim 96 \
#   --init_num_heads 2 \
#   --out_embed_dims 128 160 192 192 224 \
#   --decoding_strategy single \
#   --decoder_embed_dim 128 \
#   --decoder_depth 1 \
#   --decoder_num_heads 1
# echo "Embedding extraction completed"