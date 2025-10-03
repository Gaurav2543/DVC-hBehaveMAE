#!/bin/bash
#SBATCH --job-name=extract_embeds_optimized
#SBATCH --output=../generated_outputs/extract_embeds_optimized.out  # Save output logs
#SBATCH --error=../generated_outputs/extract_embeds_optimized.err   # Save error logs
#SBATCH --cpus-per-task=64  # Number of CPU threads
#SBATCH --partition=h100  # Change this based on your cluster’s GPU partition
#SBATCH --time=72:00:00  # Time limit (adjust as needed)
#SBATCH --gres=gpu:2  # Number of GPUs requested
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --mem=370G  # Memory requested
#SBATCH --nodes=1  # Run on a single node

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "Extracting embeddings..."
python3 downstream_tasks/extract_embeds_optimized.py \
  --dvc_root /scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final \
  --summary_csv /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv \
  --ckpt_path /scratch/bhole/dvc_data/smoothed/models_4weeks/checkpoint-09999.pth \
  --output_dir /scratch/bhole/dvc_data/smoothed/models_4weeks/extract_embeds_optimized \
  --zarr_path /scratch/bhole/dvc_data/smoothed/models_4weeks/extract_embeds_optimized_zarr \
  --time_aggregations 40320 \
  --aggregation_names 4weeks \
  --batch_size 64 \
  --multi_gpu \
  --model hbehavemae \
  --stages 2 3 4 4 5 5 6 \
  --q_strides "6,1,1;4,1,1;3,1,1;4,1,1;2,1,1;2,1,1" \
  --mask_unit_attn True False False False False False False \
  --patch_kernel 5 1 12 \
  --init_embed_dim 96 \
  --init_num_heads 2 \
  --out_embed_dims 128 160 192 192 224 224 256 \
  --decoding_strategy single \
  --decoder_embed_dim 128 \
  --decoder_depth 1 \
  --decoder_num_heads 1 
echo "Embedding extraction completed"